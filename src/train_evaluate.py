import numpy as np
from model_detection import DetectionModel
from model_cause import CauseModel
from model_prediction import PredictionModel

# ==============================================================================
# METRICS
# ==============================================================================

def calculate_classification_metrics(y_true, y_pred, threshold=0.5):
    y_pred_binary = (y_pred >= threshold).astype(float)
    
    true_positives = np.sum((y_true == 1.0) & (y_pred_binary == 1.0))
    false_positives = np.sum((y_true == 0.0) & (y_pred_binary == 1.0))
    false_negatives = np.sum((y_true == 1.0) & (y_pred_binary == 0.0))
    true_negatives = np.sum((y_true == 0.0) & (y_pred_binary == 0.0))
    
    precision = true_positives / (true_positives + false_positives + 1e-15)
    recall = true_positives / (true_positives + false_negatives + 1e-15)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-15)
    accuracy = (true_positives + true_negatives) / len(y_true)
    
    return accuracy, precision, recall, f1_score

def calculate_regression_metrics(y_true, y_pred, max_time):
    mae = np.mean(np.abs(y_true - y_pred)) * max_time
    rmse = np.sqrt(np.mean((y_true - y_pred)**2)) * max_time
    return mae, rmse

# ==============================================================================
# TRAIN_DETECTION_MODEL
# ==============================================================================

def train_detection_model(model, x_train, y_train, x_val, y_val, epochs, learning_rate):
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        indices = np.random.permutation(len(x_train))
        x_train_shuffled = x_train[indices]
        y_train_shuffled = y_train[indices]
        
        for i in range(len(x_train_shuffled)):
            x_seq = x_train_shuffled[i]
            y_t = y_train_shuffled[i]
            loss, _ = model.train_step(x_seq, y_t, learning_rate)
            epoch_loss += loss
            
        train_losses.append(epoch_loss / len(x_train))
        
        val_loss = 0.0
        for i in range(len(x_val)):
            y_p, _ = model.forward_pass(x_val[i])
            epsilon = 1e-15
            y_p_clipped = np.clip(y_p, epsilon, 1 - epsilon)
            val_loss += -(y_val[i] * np.log(y_p_clipped) + (1 - y_val[i]) * np.log(1 - y_p_clipped))
            
        val_losses.append(val_loss / len(x_val))
        
        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f}")
        
    return train_losses, val_losses

# ==============================================================================
# TRAIN_CAUSE_MODEL
# ==============================================================================

def train_cause_model(model, x_train, y_train, epochs, learning_rate):
    train_losses = []
    
    mask = y_train > 0
    x_train_causes = x_train[mask]
    y_train_causes = y_train[mask] - 1 
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        indices = np.random.permutation(len(x_train_causes))
        x_shuffled = x_train_causes[indices]
        y_shuffled = y_train_causes[indices].astype(int)
        
        for i in range(len(x_shuffled)):
            x_seq = x_shuffled[i]
            y_t_idx = y_shuffled[i]
            loss, _ = model.train_step(x_seq, y_t_idx, learning_rate)
            epoch_loss += loss
            
        train_losses.append(epoch_loss / len(x_shuffled))
        print(f"Epoch {epoch + 1}/{epochs} | Cause Train Loss: {train_losses[-1]:.4f}")
        
    return train_losses

# ==============================================================================
# TRAIN_PREDICTION_MODEL
# ==============================================================================

def train_prediction_model(model, x_train, y_train, epochs, learning_rate):
    train_losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        indices = np.random.permutation(len(x_train))
        x_shuffled = x_train[indices]
        y_shuffled = y_train[indices]
        
        for i in range(len(x_shuffled)):
            x_seq = x_shuffled[i]
            y_t = y_shuffled[i]
            loss, _ = model.train_step(x_seq, y_t, learning_rate)
            epoch_loss += loss
            
        train_losses.append(epoch_loss / len(x_train))
        print(f"Epoch {epoch + 1}/{epochs} | Pred Train Loss: {train_losses[-1]:.4f}")
        
    return train_losses