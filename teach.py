import torch
import ns
import os

from constns import *

print("--------------------------------------------------------------")
print("")


# === Параметры ===


BATCH_SIZE = 20  # размер батча
EPOCHS = 10    # количество эпох
LR = 5e-5    # скорость обучения
BETA = 0.3   # коэффициент для взвешивания потерь     


# === Нарезка окон ===
#
def create_windows(series, seq_len, pred_len):
    windows = []
    for i in range(len(series) - seq_len - pred_len):
        input_seq = series[i:i+seq_len]
        target_seq = series[i+seq_len:i+seq_len+pred_len]
        windows.append((input_seq.unsqueeze(-1), target_seq.unsqueeze(-1)))
    return windows


from data_gen import generate_crypto_like
def gen():
    return generate_crypto_like(length=1000,
                         start_price=2000,
                         max_trend=0.2,
                         max_trend_len=400,
                         max_jump=100,
                         num_jumps=5,
                         bounce1=0,
                         bounce2=0,
                         prebounce1=10,
                         prebounce2=30,
                         bounce_len=30)


# === Главный блок ===
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)


    series = ns.load_series_2(CSV_PATH)
    

    print(f"Loaded series of length {len(series)}")
    windows = create_windows(series, SEQ_LEN, PRED_LEN)
    model = ns.TimeSeriesTransformerWithAttn(input_dim=1, model_dim=MODEL_DIM, num_heads=NUM_HEADS,
                                  num_layers=NUM_LAYERS, output_dim=1)
    #model = ns.TimeSeriesTransformer(input_dim=1, model_dim=MODEL_DIM, num_heads=NUM_HEADS,
    #                              num_layers=NUM_LAYERS, output_dim=1)
    model.to(device)
    print("Model initialized.")

    #загрузить модель из файла если есть
    if "model.pth" in os.listdir():
        print("Loading model from file...")
        model.load_state_dict(torch.load("model.pth", map_location=device))
        print("Model loaded.")
    else:
        print("No model file found, training new model.")


    #загрузить optimizer из файла если есть
    if "optimizer.pth" in os.listdir():
        print("Loading optimizer from file...")
        optimizer = torch.optim.Adam(model.parameters(), LR)
        checkpoint = torch.load("optimizer.pth", map_location=device)
        optimizer.load_state_dict(checkpoint)

        print("Optimizer loaded.")
        optimizer = ns.train(model, windows, optimizer=optimizer, epochs=EPOCHS, beta = BETA, \
                             batch_size=BATCH_SIZE, lr=LR, device=device)
    else:
        print("No optimizer file found, creating new optimizer.")
        optimizer = ns.train(model, windows, epochs=EPOCHS, beta = BETA, \
                             batch_size=BATCH_SIZE, lr=LR, device=device)
    # сохранить optimizer в файл
    torch.save(optimizer.state_dict(), "optimizer.pth")

  
    print("Training completed.")

    torch.save(model.state_dict(), "model.pth")

    
