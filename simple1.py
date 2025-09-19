import numpy as np

# -------------------------------
# Простой Трансформер на numpy
# -------------------------------

# Функция позиционного кодирования (Position Encoding)
def positional_encoding(seq_len, d_model):
    """
    seq_len: длина входной последовательности
    d_model: размерность эмбеддинга
    """
    pos = np.arange(seq_len)[:, np.newaxis]  # shape (seq_len, 1)
    i = np.arange(d_model)[np.newaxis, :]    # shape (1, d_model)
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    angle_rads = pos * angle_rates

    # Чередуем синус и косинус
    pos_encoding = np.zeros((seq_len, d_model))
    pos_encoding[:, 0::2] = np.sin(angle_rads[:, 0::2])
    pos_encoding[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return pos_encoding

# Функция "Scaled Dot-Product Attention"
def scaled_dot_product_attention(Q, K, V):
    """
    Q, K, V: матрицы запросов, ключей и значений (размерности: [seq_len, d_k])
    """
    d_k = Q.shape[-1]
    scores = np.dot(Q, K.T) / np.sqrt(d_k)  # shape: (seq_len, seq_len)
    weights = softmax(scores)
    output = np.dot(weights, V)             # shape: (seq_len, d_k)
    return output, weights

def softmax(x):
    """
    Применяет softmax по последней оси
    """
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

# Класс слоя самовнимания (Self-Attention)
class SelfAttention:
    def __init__(self, d_model):
        self.d_model = d_model
        # Инициализация весов для Q, K, V
        self.W_q = np.random.randn(d_model, d_model)
        self.W_k = np.random.randn(d_model, d_model)
        self.W_v = np.random.randn(d_model, d_model)

    def __call__(self, x):
        # x: shape (seq_len, d_model)
        Q = np.dot(x, self.W_q)
        K = np.dot(x, self.W_k)
        V = np.dot(x, self.W_v)
        attn_output, attn_weights = scaled_dot_product_attention(Q, K, V)
        return attn_output

# Простейший слой feed-forward (обычный полносвязный слой)
class FeedForward:
    def __init__(self, d_model, d_ff):
        self.W1 = np.random.randn(d_model, d_ff)
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model)
        self.b2 = np.zeros(d_model)

    def __call__(self, x):
        # x: shape (seq_len, d_model)
        x = np.dot(x, self.W1) + self.b1
        x = np.maximum(0, x)  # ReLU
        x = np.dot(x, self.W2) + self.b2
        return x

# Один слой энкодера Трансформера
class TransformerEncoderLayer:
    def __init__(self, d_model, d_ff):
        self.self_attn = SelfAttention(d_model)
        self.ff = FeedForward(d_model, d_ff)

    def __call__(self, x):
        # Слой самовнимания + остаточное соединение (residual)
        attn_output = self.self_attn(x)
        x = x + attn_output  # residual
        # Слой feed-forward + residual
        ff_output = self.ff(x)
        x = x + ff_output    # residual
        return x

# Основной класс Трансформера
class SimpleTransformer:
    def __init__(self, seq_len, d_model, d_ff, num_layers):
        self.seq_len = seq_len
        self.d_model = d_model
        self.pos_encoding = positional_encoding(seq_len, d_model)
        self.layers = [TransformerEncoderLayer(d_model, d_ff) for _ in range(num_layers)]

    def encode(self, x):
        # x: shape (seq_len, d_model)
        x = x + self.pos_encoding  # добавляем позиционное кодирование
        for layer in self.layers:
            x = layer(x)
        return x

# -------------------------------
# Пример использования
# -------------------------------

# Пусть у нас есть числовой ряд (например, возрастающая последовательность)
seq = np.arange(10)  # [0, 1, 2, ..., 9]
seq_len = len(seq)
d_model = 8          # размерность эмбеддинга
d_ff = 16            # размерность скрытого слоя feed-forward
num_layers = 2       # количество слоев энкодера

# Преобразуем числа в вектора (простое расширение размерности)
# Обычно используют эмбеддинг, но для простоты просто расширим размерность
x = np.zeros((seq_len, d_model))
x[:, 0] = seq  # только первый столбец содержит значения ряда

# Создаем и применяем Трансформер
transformer = SimpleTransformer(seq_len, d_model, d_ff, num_layers)
output = transformer.encode(x)

print("Входная последовательность:")
print(seq)
print("\nВыход Трансформера (после 2 слоев):")
print(output)