import numpy as np

np.random.seed(42)

# Parámetros generales
TOTAL_TIME = 20000  # número de pasos de tiempo
MICRO_STOP_PROB = 0.003  # eventos raros (~0.3%)
MAX_LAG = 10  # desfase temporal máximo

# Variables
vibration = np.random.normal(0, 0.2, TOTAL_TIME)
current = np.random.normal(5, 0.5, TOTAL_TIME)
speed = np.random.normal(100, 2, TOTAL_TIME)
temperature = np.random.normal(70, 1, TOTAL_TIME)

# Labels
micro_stop = np.zeros(TOTAL_TIME)
cause = np.zeros(TOTAL_TIME)  # 0 = normal

# Tipos de fallas
CAUSES = {
    1: "falta_material",
    2: "desalineacion",
    3: "error_humano",
    4: "fallo_mecanico"
}

def add_noise(signal, noise_level=0.1):
    return signal + np.random.normal(0, noise_level, len(signal))

def inject_micro_stop(t):
    lag = np.random.randint(2, MAX_LAG)

    cause_type = np.random.choice(list(CAUSES.keys()))

    # Marca evento
    micro_stop[t] = 1
    cause[t] = cause_type

    # Simular causa antes (desfase)
    start = max(0, t - lag)

    if cause_type == 1:  # falta material
        speed[start:t] -= np.linspace(0, 20, t - start)
        current[start:t] -= np.linspace(0, 1, t - start)

    elif cause_type == 2:  # desalineación
        vibration[start:t] += np.linspace(0, 3, t - start)

    elif cause_type == 3:  # error humano
        speed[start:t] += np.random.normal(0, 10, t - start)

    elif cause_type == 4:  # fallo mecánico leve
        temperature[start:t] += np.linspace(0, 10, t - start)
        vibration[start:t] += np.linspace(0, 2, t - start)

    # Efecto durante micro-parada
    speed[t:t+3] = 0
    current[t:t+3] *= 0.5

# Generar eventos
for t in range(20, TOTAL_TIME - 20):
    if np.random.rand() < MICRO_STOP_PROB:
        inject_micro_stop(t)

# Añadir ruido global (simular sensores reales)
vibration = add_noise(vibration, 0.3)
current = add_noise(current, 0.2)
speed = add_noise(speed, 1.0)
temperature = add_noise(temperature, 0.5)

# Introducir datos faltantes (missing values)
def add_missing(signal, missing_prob=0.01):
    mask = np.random.rand(len(signal)) < missing_prob
    signal[mask] = np.nan
    return signal

vibration = add_missing(vibration)
current = add_missing(current)
speed = add_missing(speed)
temperature = add_missing(temperature)

# Dataset final
dataset = np.column_stack([
    vibration,
    current,
    speed,
    temperature,
    micro_stop,
    cause
])

print("Dataset shape:", dataset.shape)

# Guardar
np.savetxt("industrial_dataset.csv", dataset, delimiter=",",
           header="vibration,current,speed,temperature,micro_stop,cause",
           comments='')