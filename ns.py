import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pandas as pd
import time








# === Позиционное кодирование ===
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(1)  # <--- ВАЖНО

    def forward(self, x):
        # x: [seq_len, batch, d_model]
        x = x + self.pe[:x.size(0), :, :].to(x.device)
        return x
        
class TransformerEncoderLayerWithAttn(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiheadAttentionWithHeads(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, need_weights=False):
        src2, attn_weights = self.self_attn(src, src, src, need_weights=need_weights)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        if need_weights:
            return src, attn_weights
        return src



# === Модель трансформера ===
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(model_dim, output_dim)

    def forward(self, src):
        src = self.input_proj(src)
        src = self.pos_encoder(src)
        src = self.transformer_encoder(src)
        return self.output_proj(src)

class TimeSeriesTransformerWithAttn(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim)
        self.layers = nn.ModuleList([
            TransformerEncoderLayerWithAttn(d_model=model_dim, nhead=num_heads)
            for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(model_dim, output_dim)

    def forward(self, src, need_weights=False):
        src = self.input_proj(src)
        src = self.pos_encoder(src)
        attn_maps = []
        for layer in self.layers:
            if need_weights:
                src, attn = layer(src, need_weights=True)
                attn_maps.append(attn)  # [num_heads, seq_len, seq_len]
            else:
                src = layer(src)
        out = self.output_proj(src)
        if need_weights:
            return out, attn_maps
        return out

class MultiheadAttentionWithHeads(nn.MultiheadAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None, need_weights=True):
        # query, key, value: [tgt_len, batch, embed_dim]
        tgt_len, bsz, embed_dim = query.size()
        head_dim = embed_dim // self.num_heads
        assert head_dim * self.num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Разрезаем общий in_proj на q/k/v
        q_proj_weight, k_proj_weight, v_proj_weight = self.in_proj_weight.chunk(3, dim=0)
        if self.in_proj_bias is not None:
            q_proj_bias, k_proj_bias, v_proj_bias = self.in_proj_bias.chunk(3, dim=0)
        else:
            q_proj_bias = k_proj_bias = v_proj_bias = None

        q = F.linear(query, q_proj_weight, q_proj_bias)
        k = F.linear(key,   k_proj_weight, k_proj_bias)
        v = F.linear(value, v_proj_weight, v_proj_bias)

        # Явное масштабирование (вместо self.scaling)
        scale = 1.0 / math.sqrt(head_dim)
        q = q * scale

        # [tgt_len, batch, embed_dim] -> [batch*num_heads, tgt_len, head_dim]
        def shape(x):
            return x.contiguous().view(tgt_len, bsz, self.num_heads, head_dim) \
                     .permute(1, 2, 0, 3) \
                     .reshape(bsz * self.num_heads, tgt_len, head_dim)

        q = shape(q)
        k = shape(k).contiguous()
        v = shape(v).contiguous()

        # Attention weights: [bn, tgt_len, src_len]
        attn_output_weights = torch.bmm(q, k.transpose(1, 2))

        # Маски (если есть)
        if attn_mask is not None:
            # ожидается [tgt_len, src_len] или broadcastable к [bn, tgt_len, src_len]
            attn_output_weights = attn_output_weights + attn_mask

        if key_padding_mask is not None:
            # key_padding_mask: [batch, src_len], True = маскируем
            # расширим на num_heads
            kpm = key_padding_mask.repeat_interleave(self.num_heads, dim=0)  # [bn, src_len]
            attn_output_weights = attn_output_weights.masked_fill(kpm.unsqueeze(1), float('-inf'))

        attn_output_weights = F.softmax(attn_output_weights, dim=-1)

        # Выход внимания
        attn_output = torch.bmm(attn_output_weights, v)  # [bn, tgt_len, head_dim]

        # Обратно в [tgt_len, batch, embed_dim]
        attn_output = attn_output.reshape(bsz, self.num_heads, tgt_len, head_dim) \
                                   .permute(2, 0, 1, 3) \
                                   .reshape(tgt_len, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)

        if need_weights:
            # Вернём веса по головам: [batch, num_heads, tgt_len, src_len]
            attn_weights_heads = attn_output_weights.reshape(bsz, self.num_heads, tgt_len, -1)
            return attn_output, attn_weights_heads
        else:
            return attn_output, None




# === Загрузка и нормализация данных ===
def load_series(csv_path):
    df = pd.read_csv(csv_path)
    values = torch.tensor(df["value"].values, dtype=torch.float32)
    mean = values.mean().item()
    std = values.std().item()
    normed = (values - mean) / std
    return normed, mean, std

def load_series_2(csv_path):
    df = pd.read_csv(csv_path)
    values = torch.tensor(df["value"].values, dtype=torch.float32)
    values = (values - values.mean()) / values.std()  # нормализация
    return values

# === Нарезка окон ===
def create_windows(series, seq_len, pred_len):
    windows = []
    for i in range(len(series) - seq_len - pred_len):
        input_seq = series[i:i+seq_len]
        target_seq = series[i+1:i+seq_len+1]
        windows.append((input_seq.unsqueeze(-1), target_seq.unsqueeze(-1)))
    return windows


# === Обучение
def train(model, windows, optimizer=None, lambda_grad=0.5, epochs=20, batch_size=128, lr=5e-4):
    # Если оптимизатор не передан — создаём новый
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr)

    base_loss = nn.SmoothL1Loss(beta=0.1)  # Huber Loss

    startTime = time.time()
    for epoch in range(epochs):
        batch_losses = []
        for i in range(0, len(windows), batch_size):
            batch = windows[i:i+batch_size]
            if len(batch) < batch_size:
                continue

            inputs = torch.stack([w[0] for w in batch])  # [batch, seq_len, 1]
            targets = torch.stack([w[1] for w in batch]) # [batch, seq_len, 1]
            inputs = inputs.transpose(0, 1)              # [seq_len, batch, 1]
            targets = targets.transpose(0, 1)

            outputs = model(inputs)                      # [seq_len, batch, 1]

            # Основной loss
            loss_main = base_loss(outputs, targets)

            # Градиентный штраф (если нужен)
            # diff_outputs = outputs[1:] - outputs[:-1]
            # diff_targets = targets[1:] - targets[:-1]
            # loss_grad = nn.L1Loss()(diff_outputs, diff_targets)

            # Общий loss
            loss = loss_main # + lambda_grad * loss_grad

            optimizer.zero_grad()

            loss.backward() 
            optimizer.step()
            batch_losses.append(loss.item())

        curTime = time.time()
        print(f"Epoch {epoch}, Time: {curTime - startTime:.2f}s, Loss: {sum(batch_losses)/len(batch_losses):.4f}")
        startTime = curTime

    return optimizer  # возвращаем оптимизатор, чтобы использовать дальше


# === Прогноз ===
def predict(model, known_seq, pred_len, seq_len=120):
    model.eval()
    seq = known_seq.clone().unsqueeze(0).unsqueeze(-1)  # [1, seq_len, 1]
    seq = seq.transpose(0, 1)  # [seq_len, 1, 1]
    preds = []
    with torch.no_grad():
        for _ in range(pred_len):
            out = model(seq)
            next_val = out[-1, 0, 0]
            seq = torch.cat([seq, next_val.view(1, 1, 1)], dim=0)[-seq_len:]
            preds.append(next_val.item())
    return preds




