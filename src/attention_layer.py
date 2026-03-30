import numpy as np

# ==============================================================================
# ACTIVATION_FUNCTIONS
# ==============================================================================

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

# ==============================================================================
# ATTENTION_LAYER
# ==============================================================================

class AttentionLayer:
    
    def __init__(self, hidden_size, attention_size):
        self.hidden_size = hidden_size
        self.attention_size = attention_size
        
        limit_w = np.sqrt(6 / (hidden_size + attention_size))
        self.weights_w = np.random.uniform(-limit_w, limit_w, (attention_size, hidden_size))
        self.bias_w = np.zeros((attention_size, 1))
        
        limit_v = np.sqrt(6 / attention_size)
        self.vector_v = np.random.uniform(-limit_v, limit_v, (1, attention_size))
        
    def forward_pass(self, h_states):
        seq_length = h_states.shape[0]
        
        scores = np.zeros((seq_length, 1))
        energy_states = np.zeros((seq_length, self.attention_size, 1))
        
        for t in range(seq_length):
            h_t = h_states[t].reshape(-1, 1)
            energy = np.tanh(np.dot(self.weights_w, h_t) + self.bias_w)
            energy_states[t] = energy
            scores[t] = np.dot(self.vector_v, energy)[0, 0]
            
        attention_weights = softmax(scores)
        
        context_vector = np.zeros((self.hidden_size, 1))
        for t in range(seq_length):
            context_vector += attention_weights[t, 0] * h_states[t].reshape(-1, 1)
            
        cache = (h_states, energy_states, attention_weights, scores)
        return context_vector, cache

    def backward_pass(self, d_context_vector, cache):
        h_states, energy_states, attention_weights, scores = cache
        seq_length = h_states.shape[0]
        
        d_weights_w = np.zeros_like(self.weights_w)
        d_bias_w = np.zeros_like(self.bias_w)
        d_vector_v = np.zeros_like(self.vector_v)
        
        dh_states = np.zeros_like(h_states)
        
        d_attention_weights = np.zeros((seq_length, 1))
        for t in range(seq_length):
            h_t = h_states[t].reshape(-1, 1)
            d_attention_weights[t, 0] = np.dot(d_context_vector.T, h_t)[0, 0]
            dh_states[t] = (attention_weights[t, 0] * d_context_vector).reshape(self.hidden_size, 1)
            
        d_scores = np.zeros((seq_length, 1))
        for i in range(seq_length):
            for j in range(seq_length):
                if i == j:
                    jacobian_ij = attention_weights[i, 0] * (1 - attention_weights[i, 0])
                else:
                    jacobian_ij = -attention_weights[i, 0] * attention_weights[j, 0]
                d_scores[i, 0] += d_attention_weights[j, 0] * jacobian_ij
                
        for t in range(seq_length):
            h_t = h_states[t].reshape(-1, 1)
            energy = energy_states[t]
            
            d_vector_v += d_scores[t, 0] * energy.T
            
            d_energy = d_scores[t, 0] * self.vector_v.T
            d_pre_activation = d_energy * tanh_derivative(energy)
            
            d_weights_w += np.dot(d_pre_activation, h_t.T)
            d_bias_w += d_pre_activation
            
            dh_states[t] += np.dot(self.weights_w.T, d_pre_activation).reshape(self.hidden_size, 1)
            
        gradients = (d_weights_w, d_bias_w, d_vector_v)
        return dh_states, gradients

    def update_parameters(self, gradients, learning_rate):
        d_weights_w, d_bias_w, d_vector_v = gradients
        
        np.clip(d_weights_w, -1, 1, out=d_weights_w)
        np.clip(d_vector_v, -1, 1, out=d_vector_v)
        
        self.weights_w -= learning_rate * d_weights_w
        self.bias_w -= learning_rate * d_bias_w
        self.vector_v -= learning_rate * d_vector_v