import torch
import torch.nn as nn
import math

import pandas as pd
import matplotlib.pyplot as plt


print("--------------------------------------------------------------")
print("")


# === Параметры ===
SEQ_LEN = 200       # длина входной последовательности
PRED_LEN = 100      # сколько шагов предсказывать
MODEL_DIM = 8  # размерность модели
NUM_HEADS = 4   # количество голов в Multi-Head Attention
NUM_LAYERS = 2  # количество слоев трансформера
CSV_PATH = "data.csv"  # файл с колонками [time, value]

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

def load_series(csv_path):
    df = pd.read_csv(csv_path)
    values = torch.tensor(df["value"].values, dtype=torch.float32)
    mean = values.mean().item()
    std = values.std().item()
    normed = (values - mean) / std
    return normed, mean, std

# === Прогноз ===
def predict(model, known_seq, pred_len):
    model.eval()
    seq = known_seq.clone().unsqueeze(0).unsqueeze(-1)  # [1, seq_len, 1]
    seq = seq.transpose(0, 1)  # [seq_len, 1, 1]
    preds = []
    with torch.no_grad():
        for _ in range(pred_len):
            out = model(seq)
            next_val = out[-1, 0, 0]
            seq = torch.cat([seq, next_val.view(1, 1, 1)], dim=0)[-SEQ_LEN:]
            preds.append(next_val.item())
    return preds

# === Главный блок ===
if __name__ == "__main__":
    print("Using device:", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    # Загрузка и подготовка данных
    series, mean, std = load_series(CSV_PATH)

    
    print(f"Loaded series of length {len(series)}")
    
    model = TimeSeriesTransformer(input_dim=1, model_dim=MODEL_DIM, num_heads=NUM_HEADS,
                             num_layers=NUM_LAYERS, output_dim=1)
    model.load_state_dict(torch.load("model.pth"))
    model.eval()  # если только для инференса
    print("Model initialized.")

    # Прогноз на основе последних SEQ_LEN значений
    known = series[-SEQ_LEN:] 
    predicted2 = predict(model, known, PRED_LEN)
    predicted = [p * std + mean for p in predicted2]

    k = known * std + mean
    print ("Known values:", k)
    print ("Predicted values:", predicted)
    
    # Визуализация
    
    plt.figure(figsize=(10, 5))
    # Известные значения (синим)
    plt.plot(range(SEQ_LEN), k, label="Known", color="blue")
    # Предсказанные значения (оранжевым)
    plt.plot(range(SEQ_LEN, SEQ_LEN + PRED_LEN), predicted, label="Predicted", color="orange")
    plt.legend()
    plt.title("Currency Forecast")
    plt.xlabel("Time step")
    plt.ylabel("Value")
    plt.grid(True)
    plt.show()



