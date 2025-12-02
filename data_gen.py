import math
import random
import csv
import time
from datetime import datetime
import numpy as np


#простая синусоида
def generate_sin2(num_samples, amplitude=8, frequency=1):
    X, Y = [], []
    for k in range(num_samples):
        v = math.sin(k/180 * frequency) * amplitude
        X.append(k)
        Y.append(v)
    return X, Y


def generate_varfreq_sin(num_samples, amplitude=8, freq_start=1.0, freq_end=5.0):
    X, Y = [], []
    for k in range(num_samples):
        # Линейная интерполяция частоты от freq_start к freq_end
        progress = k / num_samples
        freq = freq_start + (freq_end - freq_start) * progress

        # Вычисляем значение синуса с текущей частотой
        v = math.sin(k / 180 * freq) * amplitude
        X.append(k)
        Y.append(v)
    return X, Y

def generate_lineal(num_samples, start_val=0, end_val=100):
    X, Y = [], []
    for k in range(num_samples):
        progress = k / num_samples
        v = start_val + (end_val - start_val) * progress
        X.append(k)
        Y.append(v)
    return X, Y

def generate_pila(num_samples, amplitude=8, freq_start=1.0, freq_end=5.0):
    X, Y = [], []
    for k in range(num_samples):
        # Линейная интерполяция частоты от freq_start к freq_end
        progress = k / num_samples
        freq = freq_start + (freq_end - freq_start) * progress

        # Вычисляем значение синуса с текущей частотой
        v = (k % freq)/freq * amplitude 
        X.append(k)
        Y.append(v)
    return X, Y

def generate_square_wave(num_samples, amp_min=0, amp_max=10, freq_min=10.0, freq_max=50.0):
    X, Y = [], []
    k = 0
    state = 0
    k_prev = 0
    while k < num_samples:
        
        freq = freq_min + (freq_max - freq_min)*random.random()
        amp = amp_min + (amp_max - amp_min) 
        if state == 1:
            state = 0
        else:
            state = 1
        v = amp*state

        while k-k_prev < freq:
            k+=1 
            X.append(k)
            Y.append(v)
        k_prev = k

    return X, Y


def add_noise(Y, noise_level=0.1):
    noisy_Y = []
    for y in Y:
        noise = random.uniform(-noise_level, noise_level)
        noisy_Y.append(y + noise)
    return noisy_Y

def generate_real_data(num_samples, file_name, start_pos):
    timestamps = []
    closes = []

    # Чтение CSV-файла
    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)

    total_rows = len(rows)
    print(f"[INFO] Всего строк в файле: {total_rows}")

    # Проверка, достаточно ли данных
    if start_pos + num_samples > total_rows:
        raise ValueError(f"Недостаточно данных: требуется {start_pos + num_samples}, доступно {total_rows}")

    # Извлечение нужных строк
    for row in rows[start_pos : start_pos + num_samples]:
        try:
            # Преобразование timestamp → Unix
            dt = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")
            unix_ts = int(time.mktime(dt.timetuple()))
            close_price = float(row[4])  # close

            timestamps.append(unix_ts)
            closes.append(close_price)
        except Exception as e:
            print(f"[WARN] Ошибка в строке: {row} → {e}")
            continue

    return timestamps, closes


def generate_test_data(total_length, num_patterns, file_names, repetitions, intersections=False):
    """
    Генерация тестовых данных с паттернами.

    Аргументы:
    - total_length: длина итоговых массивов X и Y
    - num_patterns: количество паттернов (от 1 до 8)
    - file_names: список имён файлов с паттернами
    - repetitions: сколько раз вставлять каждый паттерн
    - intersections: допустимы ли пересечения

    Возвращает:
    - X: массив индексов [0..total_length-1]
    - Y: массив данных с паттернами и константой
    """

    assert num_patterns <= len(file_names), "Недостаточно файлов для указанного числа паттернов"
    assert num_patterns <= 8, "Максимум 8 паттернов"

    # Загружаем паттерны
    patterns = []
    for fname in file_names[:num_patterns]:
        pat = np.loadtxt(fname)  # допустим, паттерн хранится в текстовом файле
        print(f"Загружен паттерн из {fname}, длина = {len(pat)}")
        patterns.append(pat)

    # X — просто индексы
    X = np.arange(total_length)

    # Y — заполняем случайной константой
    base_value = 140 #random.uniform(50, 100)
    Y = np.full(total_length, base_value)

    if not intersections:
        # Считаем общую длину всех вставок
        total_insert_len = sum(len(pat) for pat in patterns) * repetitions
        if total_insert_len > total_length:
            raise ValueError("Невозможно разместить паттерны без пересечений: слишком мало места")

    # Список занятых интервалов
    occupied = []

    # Вставляем паттерны
    for pat in patterns:
        pat_len = len(pat)
        for _ in range(repetitions):
            # ищем свободное место
            placed = False
            for _ in range(1000):  # ограничение на количество попыток
                pos = random.randint(0, total_length-1)
                interval = (pos, pos + pat_len)
                
                #проверка, чтобы паттерн не выходил за границы = если выходит, то обрезать паттерн
                if pos + pat_len > total_length:
                    pat = pat[:total_length - pos]
                    pat_len = len(pat)
                    print(f"Обрезан паттерн до длины {pat_len} для позиции {pos}")

                # проверяем пересечения, если запрещены
                if intersections or all(not (pos < end and pos+pat_len > start) for start, end in occupied):
                    Y[pos:pos+pat_len] = pat
                    occupied.append(interval)
                    placed = True
                    break
            if not placed:
                raise ValueError("Не удалось разместить паттерны без пересечений")

    return X, Y



def generate_crypto_like(length,
                         start_price=100,
                         max_trend=2,
                         max_trend_len=100,
                         max_jump=100,
                         num_jumps=5,
                         bounce1=0,
                         bounce2=0,
                         prebounce1=5,
                         prebounce2=10,
                         bounce_len=3):
    series = [start_price]
    pos = 0
    
    # --- карта скачков (случайные позиции) ---
    jump_positions = sorted(random.sample(range(20, length-50), num_jumps))
    
    while len(series) < length:
        # --- тренд ---
        trend_len = random.randint(20, max_trend_len)
        slope = random.uniform(-max_trend, max_trend)
        
        for _ in range(trend_len):
            if len(series) >= length:
                break
            series.append(series[-1] + slope)
            pos += 1
            
            # --- проверяем скачки ---
            if pos in jump_positions:
                

                jump = random.uniform(-max_jump, max_jump)
                # предотскоки
                base = series[-1]
                for b in [ prebounce1, prebounce2]:
                    b = b * jump / max_jump
                    for i in range(bounce_len):
                        if len(series) >= length:
                            break
                        k= math.pi*2/bounce_len*i
                        k2=math.sin(k) 
                        series.append(base + b * k2)
                        pos += 1
                
                
                series[-1] += jump
                
                # отскоки
                for b in [ -bounce1, +bounce1, -bounce2, +bounce2 ]:
                    for _ in range(bounce_len):
                        if len(series) >= length:
                            break
                        b = b * abs(jump) / max_jump
                        series.append(series[-1] + b / bounce_len)
                        pos += 1
    
    # --- формат выхода ---
    X = np.arange(len(series[:length]))
    Y = np.array(series[:length])
    return X, Y



# Параметры
NUM_SAMPLES = 1000
#random.seed(67) # для воспроизводимости


#X, Y = generate_sin2(num_samples=NUM_SAMPLES,frequency=5)
#X, Y = generate_varfreq_sin(num_samples=NUM_SAMPLES,freq_start=10, freq_end=50)
#X, Y = generate_pila(num_samples=NUM_SAMPLES,freq_start=10, freq_end=40)
#X, Y = generate_square_wave(num_samples=NUM_SAMPLES,amp_min=4, amp_max=4, freq_min=20, freq_max=20)
#X, Y = generate_lineal(num_samples=NUM_SAMPLES, start_val=10, end_val=130) 
# X, Y = generate_real_data(NUM_SAMPLES,"eth_minute_data_rost.csv",0)

#X, Y = generate_test_data(total_length=NUM_SAMPLES,
#                            num_patterns=1,
#                            file_names=["pattern3.txt", "pattern1.txt", "pattern3.txt", "pattern4.txt"],
#                            repetitions=20,
#                            intersections=False)



X, Y = generate_crypto_like(length=NUM_SAMPLES,
                         start_price=2000,
                         max_trend=0.2,
                         max_trend_len=400,
                         max_jump=100,
                         num_jumps=8,
                         bounce1=0,
                         bounce2=0,
                         prebounce1=10,
                         prebounce2=30,
                         bounce_len=20)


#Y = add_noise(Y, noise_level=0.9)

f = open("data.csv", "w+") 
f.write("time,value\n")
for i in range(len(X)):
    f.write(str(X[i])+", " + str(Y[i]) + "\n")
f.close()



