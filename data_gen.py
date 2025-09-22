import math

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





# Параметры
NUM_SAMPLES = 1000

#X, Y = generate_sin2(num_samples=NUM_SAMPLES,frequency=50)
X, Y = generate_varfreq_sin(num_samples=NUM_SAMPLES,freq_start=10, freq_end=50)

f = open("data.csv", "w+") 
f.write("time,value\n")
for i in range(len(X)):
    f.write(str(X[i])+", " + str(Y[i]) + "\n")
f.close()
