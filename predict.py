import torch
import matplotlib.pyplot as plt
import ns
import numpy as np
from constns import *

print("--------------------------------------------------------------")
print("")


# === Параметры ===

START_POINT = 0  #начальная позиция входных данных
WINDOW = 1000 # какой кусок данных берем для рисунка графика и предсказания


# === Главный блок ===
if __name__ == "__main__":
    # Загрузка и подготовка данных
    series, mean, std = ns.load_series(CSV_PATH)

    
    print(f"Loaded series of length {len(series)}") 
    
    model = ns.TimeSeriesTransformerWithAttn(input_dim=1, model_dim=MODEL_DIM, num_heads=NUM_HEADS,
                             num_layers=NUM_LAYERS, output_dim=1)
    model.load_state_dict(torch.load("model.pth"))
    model.eval()
    print("Model initialized.")
  
    # Прогноз на основе последних SEQ_LEN значений
    graph1 = series[START_POINT:START_POINT+WINDOW]  # исторические данные для графика
    graph2 = graph1.clone()   # предсказанные данные для графика

    for i in range(0,WINDOW-SEQ_LEN-PRED_LEN,PRED_LEN):
        known = graph1[i:i+SEQ_LEN]       
        #predict = torch.tensor(ns.predict(model, known, PRED_LEN))     # предсказываем !!!
        predict = torch.tensor(ns.predict_block(model, known, PRED_LEN))     # предсказываем !!!
        graph2[i+SEQ_LEN:i+SEQ_LEN+PRED_LEN] = predict  # сохраняем в график предсказаний
      
    graph1 = graph1.numpy() * std + mean  # денормализация
    graph2 = graph2.numpy() * std + mean  # денормализация
        
   

# Включаем тёмный стиль
plt.style.use("dark_background")

# Визуализация
plt.figure(figsize=(10, 5))

# Исторические данные
plt.plot(range(len(graph1)), graph1, label="Исторические данные", color="cyan", linewidth=1)

# Предсказания
plt.plot(range(len(graph2)), graph2, label="Предсказания модели", color="orange", linewidth=1)



# Настройки
plt.legend(facecolor="black", edgecolor="white")
plt.title("Currency Forecast", color="white")
plt.xlabel("Time step", color="white")
plt.ylabel("Value", color="white")
step = 50
plt.xticks(np.arange(0, len(graph1), step))
plt.grid(True, color="gray", alpha=0.3)
plt.tight_layout()
plt.show()

