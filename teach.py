import torch
import matplotlib.pyplot as plt
import ns

print("--------------------------------------------------------------")
print("")


# === Параметры ===

SEQ_LEN = 120       # длина входной последовательности
PRED_LEN = 20      # сколько шагов предсказывать
MODEL_DIM = 64  # размерность модели
NUM_HEADS = 8   # количество голов в Multi-Head Attention
NUM_LAYERS = 3  # количество слоев трансформера
CSV_PATH = "data.csv"  # файл с колонками [time, value]

BATCH_SIZE = 128  # размер батча
EPOCHS = 20    # количество эпох
LR = 5e-4    # скорость обучения





def plot_attention(attn_maps, layer=0, head=0):
    """
    attn_maps: список attention карт от всех слоёв
    layer: индекс слоя
    head: индекс головы
    """
    attn = attn_maps[layer][head].detach().cpu().numpy()  # [seq_len, seq_len]
    plt.figure(figsize=(6,6))
    plt.imshow(attn, cmap="viridis")
    plt.title(f"Attention Layer {layer}, Head {head}")
    plt.xlabel("Key positions")
    plt.ylabel("Query positions")
    plt.colorbar()
    plt.show()

        

# === Главный блок ===
if __name__ == "__main__":
    print("Using device:", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    series = ns.load_series(CSV_PATH)
    
    print(f"Loaded series of length {len(series)}")
    windows = ns.create_windows(series, SEQ_LEN, PRED_LEN)
    model = ns.TimeSeriesTransformerWithAttn(input_dim=1, model_dim=MODEL_DIM, num_heads=NUM_HEADS,
                                  num_layers=NUM_LAYERS, output_dim=1)
    print("Model initialized.")
    ns.train(model, windows, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR)
    print("Training completed.")

    torch.save(model.state_dict(), "model.pth")

    
