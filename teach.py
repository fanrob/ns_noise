import torch
import ns
import os

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
LR = 5e-3    # скорость обучения
     

# === Главный блок ===
if __name__ == "__main__":
    print("Using device:", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    series = ns.load_series_2(CSV_PATH)
    
    print(f"Loaded series of length {len(series)}")
    windows = ns.create_windows(series, SEQ_LEN, PRED_LEN)
    model = ns.TimeSeriesTransformerWithAttn(input_dim=1, model_dim=MODEL_DIM, num_heads=NUM_HEADS,
                                  num_layers=NUM_LAYERS, output_dim=1)
    print("Model initialized.")
    #ns.train(model, windows, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR)

    #загрузить optimizer из файла если есть
    if "optimizer.pth" in os.listdir():
        print("Loading optimizer from file...")
        optimizer = torch.optim.Adam(model.parameters(), LR)
        checkpoint = torch.load("optimizer.pth")
        optimizer.load_state_dict(checkpoint)  
        print("Optimizer loaded.")
        ns.train(model, windows, optimizer=optimizer, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR)
    else:
        print("No optimizer file found, creating new optimizer.")
        optimizer = ns.train(model, windows, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR)
    # сохранить optimizer в файл
    torch.save(optimizer.state_dict(), "optimizer.pth")

  
    print("Training completed.")

    torch.save(model.state_dict(), "model.pth")

    
