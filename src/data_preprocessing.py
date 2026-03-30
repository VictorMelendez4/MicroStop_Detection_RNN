import numpy as np

# ========================================
# CONSTANTES
# ========================================

WINDOW_SIZE = 20
FEATURES = ['vibration', 'current', 'speed', 'temperature']
N_FEATURES = 4
RANDOM_SEED = 42
SMOTE_NEIGHBORS = 3

# ========================================
# CARGA DE DATOS
# ========================================

def load_raw_data(file_path):
    data = np.genfromtxt(
        file_path,
        delimiter=',',
        skip_header=1,
        filling_values=np.nan
    )
    x_raw = data[:, :4]
    y_stop = data[:, 4].astype(float)
    y_cause = data[:, 5].astype(float)
    return x_raw, y_stop, y_cause

# ========================================
# IMPUTACIÓN
# ========================================

def compute_column_medians(x):
    medians = np.zeros(x.shape[1])
    for col in range(x.shape[1]):
        col_data = x[:, col]
        valid = col_data[~np.isnan(col_data)]
        medians[col] = np.median(valid)
    return medians

def impute_with_medians(x, medians):
    x_imputed = x.copy()
    for col in range(x.shape[1]):
        nan_mask = np.isnan(x_imputed[:, col])
        x_imputed[nan_mask, col] = medians[col]
    return x_imputed

# ========================================
# CLIPPING DE OUTLIERS
# ========================================

def clip_outliers(x, lower_percentile=1, upper_percentile=99):
    x_clipped = x.copy()
    bounds = np.zeros((x.shape[1], 2))
    for col in range(x.shape[1]):
        lower = np.percentile(x_clipped[:, col], lower_percentile)
        upper = np.percentile(x_clipped[:, col], upper_percentile)
        x_clipped[:, col] = np.clip(x_clipped[:, col], lower, upper)
        bounds[col] = [lower, upper]
    return x_clipped, bounds

# ========================================
# NORMALIZACIÓN
# ========================================

def compute_normalization_params(x):
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std[std == 0] = 1.0
    return mean, std

def normalize(x, mean, std):
    return (x - mean) / std

# ========================================
# CONSTRUCCIÓN DE VENTANAS TEMPORALES
# ========================================

def build_windows(x, y_stop, y_cause, window_size=WINDOW_SIZE):
    n_samples = x.shape[0] - window_size
    x_windows = np.zeros((n_samples, window_size, x.shape[1]))
    y_stop_windowed = np.zeros(n_samples)
    y_cause_windowed = np.zeros(n_samples)

    for i in range(n_samples):
        x_windows[i] = x[i : i + window_size]
        y_stop_windowed[i] = y_stop[i + window_size]
        y_cause_windowed[i] = y_cause[i + window_size]

    return x_windows, y_stop_windowed, y_cause_windowed

# ========================================
# ETIQUETAS PARA PREDICCIÓN TEMPORAL
# ========================================

def build_time_to_next_stop(y_stop, window_size=WINDOW_SIZE):
    n = len(y_stop)
    time_to_stop = np.full(n, np.nan)

    for i in range(n):
        if y_stop[i] == 1.0:
            time_to_stop[i] = 0.0
        else:
            for future in range(i + 1, n):
                if y_stop[future] == 1.0:
                    time_to_stop[i] = float(future - i)
                    break

    valid_mask = ~np.isnan(time_to_stop)
    time_to_stop_valid = time_to_stop[valid_mask]
    max_time = np.max(time_to_stop_valid) if len(time_to_stop_valid) > 0 else 1.0
    time_to_stop_norm = np.where(valid_mask, time_to_stop / max_time, 0.0)

    windowed = time_to_stop_norm[window_size:]
    return windowed, max_time

# ========================================
# OVERSAMPLE MANUAL (SMOTE SIMPLIFICADO)
# ========================================

def _euclidean_distance(a, b):
    diff = a.reshape(-1) - b.reshape(-1)
    return np.sqrt(np.dot(diff, diff))

def smote_oversample(x_windows, y_stop, y_cause, y_time, seed=RANDOM_SEED, k=SMOTE_NEIGHBORS):
    rng = np.random.default_rng(seed)

    minority_mask = y_stop == 1.0
    x_min = x_windows[minority_mask]
    y_cause_min = y_cause[minority_mask]
    y_time_min = y_time[minority_mask]

    n_minority = x_min.shape[0]
    n_majority = np.sum(y_stop == 0.0)
    n_synthetic = n_majority - n_minority

    x_min_flat = x_min.reshape(n_minority, -1)

    synthetic_x = []
    synthetic_cause = []
    synthetic_time = []

    for _ in range(n_synthetic):
        idx = rng.integers(0, n_minority)
        sample_flat = x_min_flat[idx]

        distances = np.array([
            _euclidean_distance(sample_flat, x_min_flat[j])
            for j in range(n_minority) if j != idx
        ])
        neighbor_indices = np.argsort(distances)[:k]
        neighbor_idx = neighbor_indices[rng.integers(0, k)]
        if neighbor_idx >= idx:
            neighbor_idx += 1

        neighbor_flat = x_min_flat[neighbor_idx]
        alpha = rng.uniform(0, 1)
        new_sample_flat = sample_flat + alpha * (neighbor_flat - sample_flat)
        new_sample = new_sample_flat.reshape(x_min.shape[1], x_min.shape[2])

        synthetic_x.append(new_sample)
        synthetic_cause.append(y_cause_min[idx])
        synthetic_time.append(y_time_min[idx])

    synthetic_x = np.array(synthetic_x)
    synthetic_cause = np.array(synthetic_cause)
    synthetic_time = np.array(synthetic_time)
    synthetic_stop = np.ones(n_synthetic)

    x_balanced = np.concatenate([x_windows, synthetic_x], axis=0)
    y_stop_balanced = np.concatenate([y_stop, synthetic_stop], axis=0)
    y_cause_balanced = np.concatenate([y_cause, synthetic_cause], axis=0)
    y_time_balanced = np.concatenate([y_time, synthetic_time], axis=0)

    shuffle_idx = rng.permutation(len(y_stop_balanced))
    return (
        x_balanced[shuffle_idx],
        y_stop_balanced[shuffle_idx],
        y_cause_balanced[shuffle_idx],
        y_time_balanced[shuffle_idx]
    )

# ========================================
# SPLIT TRAIN / VAL / TEST
# ========================================

def temporal_split(x, y_stop, y_cause, y_time, train_ratio=0.7, val_ratio=0.15):
    n = len(y_stop)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    x_train = x[:train_end]
    x_val   = x[train_end:val_end]
    x_test  = x[val_end:]

    y_stop_train = y_stop[:train_end]
    y_stop_val   = y_stop[train_end:val_end]
    y_stop_test  = y_stop[val_end:]

    y_cause_train = y_cause[:train_end]
    y_cause_val   = y_cause[train_end:val_end]
    y_cause_test  = y_cause[val_end:]

    y_time_train = y_time[:train_end]
    y_time_val   = y_time[train_end:val_end]
    y_time_test  = y_time[val_end:]

    return (
        x_train, x_val, x_test,
        y_stop_train, y_stop_val, y_stop_test,
        y_cause_train, y_cause_val, y_cause_test,
        y_time_train, y_time_val, y_time_test
    )

# ========================================
# PIPELINE COMPLETO
# ========================================

def preprocess_pipeline(file_path):
    x_raw, y_stop, y_cause = load_raw_data(file_path)

    medians = compute_column_medians(x_raw)
    x_imputed = impute_with_medians(x_raw, medians)

    x_clipped, clip_bounds = clip_outliers(x_imputed)

    mean, std = compute_normalization_params(x_clipped)
    x_norm = normalize(x_clipped, mean, std)

    x_windows, y_stop_w, y_cause_w = build_windows(x_norm, y_stop, y_cause)
    y_time_w, max_time = build_time_to_next_stop(y_stop, window_size=WINDOW_SIZE)

    min_len = min(len(x_windows), len(y_time_w))
    x_windows  = x_windows[:min_len]
    y_stop_w   = y_stop_w[:min_len]
    y_cause_w  = y_cause_w[:min_len]
    y_time_w   = y_time_w[:min_len]

    splits = temporal_split(x_windows, y_stop_w, y_cause_w, y_time_w)
    (
        x_train, x_val, x_test,
        y_stop_train, y_stop_val, y_stop_test,
        y_cause_train, y_cause_val, y_cause_test,
        y_time_train, y_time_val, y_time_test
    ) = splits

    (
        x_train_bal,
        y_stop_train_bal,
        y_cause_train_bal,
        y_time_train_bal
    ) = smote_oversample(x_train, y_stop_train, y_cause_train, y_time_train)

    preprocessing_params = {
        'medians': medians,
        'clip_bounds': clip_bounds,
        'mean': mean,
        'std': std,
        'max_time': max_time,
        'window_size': WINDOW_SIZE
    }

    return (
        x_train_bal, x_val, x_test,
        y_stop_train_bal, y_stop_val, y_stop_test,
        y_cause_train_bal, y_cause_val, y_cause_test,
        y_time_train_bal, y_time_val, y_time_test,
        preprocessing_params
    )