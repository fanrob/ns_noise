import torch
import matplotlib.pyplot as plt
import ns
from matplotlib.widgets import Slider

print("--------------------------------------------------------------")
print("")


# === Параметры ===

SEQ_LEN = 120       # длина входной последовательности
PRED_LEN = 5      # сколько шагов предсказывать
MODEL_DIM = 128  # размерность модели
NUM_HEADS = 8   # количество голов в Multi-Head Attention
NUM_LAYERS = 3  # количество слоев трансформера
CSV_PATH = "data.csv"  # файл с колонками [time, value]


     

def plot_with_sliders(attn_maps):
    layer, head = 0, 0
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    img = ax.imshow(attn_maps[layer][0, head].detach().cpu().numpy(), cmap="viridis")
    ax.set_title(f"Layer {layer}, Head {head}")
    plt.colorbar(img, ax=ax)

    ax_layer = plt.axes([0.25, 0.1, 0.65, 0.03])
    ax_head = plt.axes([0.25, 0.05, 0.65, 0.03])

    slider_layer = Slider(ax_layer, 'Layer', 0, len(attn_maps)-1, valinit=0, valstep=1)
    slider_head = Slider(ax_head, 'Head', 0, NUM_HEADS-1, valinit=0, valstep=1)

    def update(val):
        l = int(slider_layer.val)
        h = int(slider_head.val)
        img.set_data(attn_maps[l][0, h].detach().cpu().numpy())
        ax.set_title(f"Layer {l}, Head {h}")
        fig.canvas.draw_idle()

    slider_layer.on_changed(update)
    slider_head.on_changed(update)

    plt.show()



def plot_input_proj_weights(model):
    weights = model.input_proj.weight.detach().cpu().numpy()  # [model_dim, input_dim]
    fig, ax = plt.subplots()
    im = ax.imshow(weights, cmap="coolwarm", aspect="auto")
    ax.set_title("Input Projection Weights")
    ax.set_xlabel("Input Features")
    ax.set_ylabel("Model Dimensions")
    plt.colorbar(im, ax=ax)
    plt.show()



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
    

    inputs = torch.randn(SEQ_LEN, 1, 1)
    out, attn_maps = model(inputs, need_weights=True)   

    
    # вызов
    #plot_with_sliders(attn_maps)
    plot_input_proj_weights(model)

    quit()
