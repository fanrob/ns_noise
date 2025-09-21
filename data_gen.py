import math

def generate_sin2(num_samples, amplitude=8, frequency=1):
    X, Y = [], []
    for k in range(num_samples):
        v = math.sin(k/180 * frequency) * amplitude
        X.append(k)
        Y.append(v)
    return X, Y

# Параметры
NUM_SAMPLES = 10000

X, Y = generate_sin2(NUM_SAMPLES)

f = open("data.csv", "w+") 
for i in range(len(X)):
    f.write(str(X[i])+", " + str(Y[i]) + "\n")
f.close()
