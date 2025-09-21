import torch
import torch.nn as nn
import math

import pandas as pd
import time
import matplotlib.pyplot as plt

# Загрузка данных из CSV
# def load_data(file_path):
#     data = open(file_path, 'r')

#     X, Y = [], []
#     for s in data:
#         x, y = s.strip().split(',')
#         X.append(float(x))
#         Y.append(float(y))
        
#     data.close()
#     return X,Y

# # Параметры
# SEQ_LEN = 30        # Длина входной последовательности
# PRED_LEN = 5        # Длина предсказания
# NUM_SAMPLES = 1000  # Количество обучающих примеров
# BATCH_SIZE = 32     # Размер батча
# EPOCHS = 50         # Количество эпох
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# print("Using " + DEVICE)
#X,Y = load_data("data.csv")

print("--------------------------------------------------------------")
print("")


# === Параметры ===

SEQ_LEN = 100       # длина входной последовательности
PRED_LEN = 30      # сколько шагов предсказывать
MODEL_DIM = 8  # размерность модели
NUM_HEADS = 4   # количество голов в Multi-Head Attention
NUM_LAYERS = 2  # количество слоев трансформера
CSV_PATH = "data.csv"  # файл с колонками [time, value]

BATCH_SIZE = 256 # размер батча
EPOCHS = 10    # количество эпох
LR = 1e-4    # скорость обучения


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

# === Загрузка и нормализация данных ===
def load_series(csv_path):
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

# === Обучение ===
def train(model, windows):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    startTime = time.time()
    for epoch in range(EPOCHS):
        batch_losses = []
        for i in range(0, len(windows), BATCH_SIZE):
            batch = windows[i:i+BATCH_SIZE]
            if len(batch) < BATCH_SIZE:
                continue
            inputs = torch.stack([w[0] for w in batch])  # [batch, seq_len, 1]
            targets = torch.stack([w[1] for w in batch]) # [batch, seq_len, 1]
            inputs = inputs.transpose(0, 1)  # [seq_len, batch, 1]
            targets = targets.transpose(0, 1)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        curTime = time.time()
        
        print(f"Epoch {epoch}, Time: {curTime - startTime:.2f}s, Loss: {sum(batch_losses)/len(batch_losses):.4f}")
        startTime = curTime
        

# === Главный блок ===
if __name__ == "__main__":
    print("Using device:", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    series = load_series(CSV_PATH)
    
    print(f"Loaded series of length {len(series)}")
    windows = create_windows(series, SEQ_LEN, PRED_LEN)
    model = TimeSeriesTransformer(input_dim=1, model_dim=MODEL_DIM, num_heads=NUM_HEADS,
                                  num_layers=NUM_LAYERS, output_dim=1)
    print("Model initialized.")
    train(model, windows)
    print("Training completed.")

    torch.save(model.state_dict(), "model.pth")


