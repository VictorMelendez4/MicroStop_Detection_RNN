import numpy as np
from lstm_cell import LstmCell
from attention_layer import AttentionLayer

# ==============================================================================
# LOSS_FUNCTION
# ==============================================================================

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# ==============================================================================
# PREDICTION_MODEL
# ==============================================================================

class PredictionModel:
    
    def __init__(self, input_size, hidden_size, attention_size):
        self.lstm = LstmCell(input_size, hidden_size)
        self.attention = AttentionLayer(hidden_size, attention_size)
        
        limit_dense = np.sqrt(6 / (hidden_size + 1))
        self.weights_dense = np.random.uniform(-limit_dense, limit_dense, (1, hidden_size))
        self.bias_dense = np.zeros((1, 1))

    def forward_pass(self, x_sequence):
        h_states, lstm_cache = self.lstm.forward_pass(x_sequence)
        context_vector, attention_cache = self.attention.forward_pass(h_states)
        
        y_pred_matrix = np.dot(self.weights_dense, context_vector) + self.bias_dense
        y_pred = y_pred_matrix[0, 0]
        
        cache = (lstm_cache, attention_cache, context_vector, y_pred_matrix)
        return y_pred, cache

    def backward_pass(self, y_true, cache):
        lstm_cache, attention_cache, context_vector, y_pred_matrix = cache
        
        dz = np.array([[y_pred_matrix[0, 0] - y_true]])
        
        d_weights_dense = np.dot(dz, context_vector.T)
        d_bias_dense = dz
        
        d_context_vector = np.dot(self.weights_dense.T, dz)
        
        dh_states, attention_gradients = self.attention.backward_pass(d_context_vector, attention_cache)
        dx_sequence, lstm_gradients = self.lstm.backward_pass(dh_states, lstm_cache)
        
        gradients = (d_weights_dense, d_bias_dense, attention_gradients, lstm_gradients)
        return gradients

    def update_parameters(self, gradients, learning_rate):
        d_weights_dense, d_bias_dense, attention_gradients, lstm_gradients = gradients
        
        np.clip(d_weights_dense, -1, 1, out=d_weights_dense)
        np.clip(d_bias_dense, -1, 1, out=d_bias_dense)
        
        self.weights_dense -= learning_rate * d_weights_dense
        self.bias_dense -= learning_rate * d_bias_dense
        
        self.attention.update_parameters(attention_gradients, learning_rate)
        self.lstm.update_parameters(lstm_gradients, learning_rate)

    def train_step(self, x_sequence, y_true, learning_rate):
        y_pred, cache = self.forward_pass(x_sequence)
        loss = mean_squared_error(y_true, y_pred)
        gradients = self.backward_pass(y_true, cache)
        self.update_parameters(gradients, learning_rate)
        return loss, y_pred