import numpy as np
from lstm_cell import LstmCell
from attention_layer import AttentionLayer

# ==============================================================================
# ACTIVATION_AND_LOSS
# ==============================================================================

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

def categorical_cross_entropy(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.sum(y_true * np.log(y_pred))

# ==============================================================================
# CAUSE_MODEL
# ==============================================================================

class CauseModel:
    
    def __init__(self, input_size, hidden_size, attention_size, num_classes=4):
        self.num_classes = num_classes
        self.lstm = LstmCell(input_size, hidden_size)
        self.attention = AttentionLayer(hidden_size, attention_size)
        
        limit_dense = np.sqrt(6 / (hidden_size + num_classes))
        self.weights_dense = np.random.uniform(-limit_dense, limit_dense, (num_classes, hidden_size))
        self.bias_dense = np.zeros((num_classes, 1))

    def forward_pass(self, x_sequence):
        h_states, lstm_cache = self.lstm.forward_pass(x_sequence)
        context_vector, attention_cache = self.attention.forward_pass(h_states)
        
        z = np.dot(self.weights_dense, context_vector) + self.bias_dense
        y_pred = softmax(z)
        
        cache = (lstm_cache, attention_cache, context_vector, y_pred)
        return y_pred, cache

    def backward_pass(self, y_true, cache):
        lstm_cache, attention_cache, context_vector, y_pred = cache
        
        dz = y_pred - y_true
        
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

    def train_step(self, x_sequence, y_true_idx, learning_rate):
        y_true = np.zeros((self.num_classes, 1))
        y_true[y_true_idx] = 1.0
        
        y_pred, cache = self.forward_pass(x_sequence)
        loss = categorical_cross_entropy(y_true, y_pred)
        gradients = self.backward_pass(y_true, cache)
        self.update_parameters(gradients, learning_rate)
        return loss, y_pred