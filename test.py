import torch
import numpy as np

import torch.nn as nn
import torch.optim as optim
import math
import pandas as pd
import matplotlib.pyplot as plt

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)


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

print("--------------------------------")
TEST = 2


if TEST == 1:
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))
    print(torch.version.cuda)




if TEST == 2:
    for bs in [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]:
        try:
            x = torch.randn(bs, 30, 1).transpose(0, 1).cuda()
            model = TimeSeriesTransformer(1, 64, 4, 2, 1).cuda()
            y = model(x)
            print(f"BATCH_SIZE {bs} — OK")
        except RuntimeError as e:
            print(f"BATCH_SIZE {bs} — FAIL: {e}")