import torch
import matplotlib.pyplot as plt
import ns
from matplotlib.widgets import Slider
from constns import *

from matplotlib.widgets import Slider
import matplotlib.pyplot as plt

def plot_with_sliders3(attn_maps, num_heads, input_seq, output_seq):
    """
    attn_maps: attention maps [layers][batch][heads][seq_len, seq_len]
    num_heads: количество attention голов
    input_seq: 1D массив входных значений
    output_seq: 1D массив выходных значений (предсказания)
    """
    layer, head = 0, 0

    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 12, 1], hspace=0.4)

    # === Верхняя часть: вход и выход на одной оси ===
    ax_seq = fig.add_subplot(gs[0, 0])
    ax_seq.plot(input_seq, color="blue", label="Входная последовательность")
    ax_seq.plot(range(len(input_seq), len(input_seq) + len(output_seq)),
                output_seq, color="orange", label="Выходная последовательность")
    ax_seq.set_title("Вход и выход")
    ax_seq.set_ylabel("Значение")
    ax_seq.legend()
    ax_seq.grid(True)

    # === Основная карта внимания ===
    ax_attn = fig.add_subplot(gs[1, 0])
    img = ax_attn.imshow(attn_maps[layer][0, head].detach().cpu().numpy(),
                         cmap="viridis")
    ax_attn.set_title(f"Layer {layer}, Head {head}")
    plt.colorbar(img, ax=ax_attn)
    ax_attn.set_xlabel("Input position")
    ax_attn.set_ylabel("Output position")

    # === Слайдеры ===
    ax_layer = plt.axes([0.25, 0.12, 0.5, 0.02])
    ax_head = plt.axes([0.25, 0.08, 0.5, 0.02])

    slider_layer = Slider(ax_layer, 'Layer', 0, len(attn_maps)-1, valinit=0, valstep=1)
    slider_head = Slider(ax_head, 'Head', 0, num_heads-1, valinit=0, valstep=1)

    def update(val):
        l = int(slider_layer.val)
        h = int(slider_head.val)
        img.set_data(attn_maps[l][0, h].detach().cpu().numpy())
        ax_attn.set_title(f"Layer {l}, Head {h}")
        fig.canvas.draw_idle()

    slider_layer.on_changed(update)
    slider_head.on_changed(update)

    plt.show()



print("--------------------------------------------------------------")
print("")



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

def plot_with_sliders2(attn_maps, num_heads):
    layer, head = 0, 0
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    img = ax.imshow(attn_maps[layer][0, head].detach().cpu().numpy(),
                    cmap="viridis")
    ax.set_title(f"Layer {layer}, Head {head}")
    plt.colorbar(img, ax=ax)

    ax_layer = plt.axes([0.25, 0.1, 0.65, 0.03])
    ax_head = plt.axes([0.25, 0.05, 0.65, 0.03])

    slider_layer = Slider(ax_layer, 'Layer', 0, len(attn_maps)-1, valinit=0, valstep=1)
    slider_head = Slider(ax_head, 'Head', 0, num_heads-1, valinit=0, valstep=1)

    def update(val):
        l = int(slider_layer.val)
        h = int(slider_head.val)
        img.set_data(attn_maps[l][0, h].detach().cpu().numpy())
        ax.set_title(f"Layer {l}, Head {h}")
        fig.canvas.draw_idle()

    slider_layer.on_changed(update)
    slider_head.on_changed(update)

    ax.set_xlabel("Input position")
    ax.set_ylabel("Output position")

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
    model.eval()
    print("Model initialized.")
    
    s = series[0:1000]
    inputs = s[0:0+SEQ_LEN] 

    # подготовка входа
    seq = inputs.clone().unsqueeze(0).unsqueeze(-1)  # [1, seq_len, 1]
    seq = seq.transpose(0, 1)  # [seq_len, 1, 1]

    with torch.no_grad():
        out,attn_maps = model(seq, need_weights=True)  # [seq_len, batch, dim]
        # берём последние pred_len шагов
        preds = out[-PRED_LEN:, 0, 0].cpu().numpy()


    
    # вызов
    #plot_with_sliders(attn_maps)
    #plot_with_sliders2(attn_maps, NUM_HEADS)
    #plot_input_proj_weights(model)
    plot_with_sliders3(attn_maps, NUM_HEADS, inputs, preds)
    

    quit()
