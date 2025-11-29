import torch
import matplotlib.pyplot as plt
import ns

print("--------------------------------------------------------------")
print("")


# === Параметры ===


SEQ_LEN = 120       # длина входной последовательности
PRED_LEN = 5      # сколько шагов предсказывать
MODEL_DIM = 64  # размерность модели
NUM_HEADS = 4   # количество голов в Multi-Head Attention
NUM_LAYERS = 4  # количество слоев трансформера
CSV_PATH = "data.csv"  # файл с колонками [time, value]

BATCH_SIZE = 256  # размер батча
EPOCHS = 5    # количество эпох
LR = 5e-2    # скорость обучения
     
     
END_POINT = -130 #конечная позиция входных данных со знаком -



# === Главный блок ===
if __name__ == "__main__":
    print("Using device:", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    # Загрузка и подготовка данных
    series, mean, std = ns.load_series(CSV_PATH)

    
    print(f"Loaded series of length {len(series)}")
    
    model = ns.TimeSeriesTransformerWithAttn(input_dim=1, model_dim=MODEL_DIM, num_heads=NUM_HEADS,
                             num_layers=NUM_LAYERS, output_dim=1)
    model.load_state_dict(torch.load("model.pth"))
    model.eval()  # если только для инференса
    print("Model initialized.")
  
    # Прогноз на основе последних SEQ_LEN значений
    point = END_POINT
    known = series[point-SEQ_LEN*2:point-SEQ_LEN]       # предпоследний кусочек длиной SEQ_LEN - 
    predicted_raw = ns.predict(model, known, PRED_LEN)     # предсказываем !!!
    
    #приведение к оригинальному масштабу
    h = series[point-SEQ_LEN*8:point]  * std + mean     # исторические данные
    k = known * std + mean                              # это шло на вход нейронки
    p = [pr * std + mean for pr in predicted_raw]       # предсказанные данные

    

    print ("Known values:", k)
    print ("Predicted values:", p)
    
    # Визуализация
    
    plt.figure(figsize=(10, 5))

    # Исторические данные — от начала
    plt.plot(range(len(h)), h, label="Исторические данные", color="black", linewidth=1)

    # Входные данные — начинаются после h[-SEQ_LEN*2:]
    offset_k = len(h) - SEQ_LEN*2
    plt.plot(range(offset_k, offset_k + SEQ_LEN), k, label="Входные данные", color="blue", linewidth=1)

    # Предсказания — начинаются после входных
    offset_p = offset_k + SEQ_LEN
    plt.plot(range(offset_p, offset_p + PRED_LEN), p, label="Предсказания модели", color="orange", linewidth=1)

    plt.legend()
    plt.title("Currency Forecast")
    plt.xlabel("Time step")
    plt.ylabel("Value")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    
