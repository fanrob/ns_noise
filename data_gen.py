import math
import random
import csv
import time
from datetime import datetime

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






# Параметры
NUM_SAMPLES = 1000

#X, Y = generate_sin2(num_samples=NUM_SAMPLES,frequency=5)
#X, Y = generate_varfreq_sin(num_samples=NUM_SAMPLES,freq_start=10, freq_end=50)
#X, Y = generate_pila(num_samples=NUM_SAMPLES,freq_start=10, freq_end=40)
#X, Y = generate_square_wave(num_samples=NUM_SAMPLES,amp_min=4, amp_max=4, freq_min=20, freq_max=20)
X, Y = generate_real_data(NUM_SAMPLES,"eth_minute_data_rost.csv",2000)


#Y2 = add_noise(Y, noise_level=0.1)

f = open("data-x.csv", "w+") 
f.write("time,value\n")
for i in range(len(X)):
    f.write(str(X[i])+", " + str(Y[i]) + "\n")
f.close()



