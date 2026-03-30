import numpy as np

# ==============================================================================
# ACTIVATION_FUNCTIONS
# ==============================================================================

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

# ==============================================================================
# LSTM_CELL
# ==============================================================================

class LstmCell:
    
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        concat_size = input_size + hidden_size
        
        limit_f = np.sqrt(6 / (concat_size + hidden_size))
        self.weights_f = np.random.uniform(-limit_f, limit_f, (hidden_size, concat_size))
        self.bias_f = np.ones((hidden_size, 1)) 
        
        limit_i = np.sqrt(6 / (concat_size + hidden_size))
        self.weights_i = np.random.uniform(-limit_i, limit_i, (hidden_size, concat_size))
        self.bias_i = np.zeros((hidden_size, 1))
        
        limit_c = np.sqrt(6 / (concat_size + hidden_size))
        self.weights_c = np.random.uniform(-limit_c, limit_c, (hidden_size, concat_size))
        self.bias_c = np.zeros((hidden_size, 1))
        
        limit_o = np.sqrt(6 / (concat_size + hidden_size))
        self.weights_o = np.random.uniform(-limit_o, limit_o, (hidden_size, concat_size))
        self.bias_o = np.zeros((hidden_size, 1))

    def forward_pass(self, x_sequence):
        seq_length = x_sequence.shape[0]
        
        h_states = np.zeros((seq_length + 1, self.hidden_size, 1))
        c_states = np.zeros((seq_length + 1, self.hidden_size, 1))
        
        f_gates = np.zeros((seq_length, self.hidden_size, 1))
        i_gates = np.zeros((seq_length, self.hidden_size, 1))
        c_bar_gates = np.zeros((seq_length, self.hidden_size, 1))
        o_gates = np.zeros((seq_length, self.hidden_size, 1))
        
        for t in range(seq_length):
            x_t = x_sequence[t].reshape(-1, 1)
            h_prev = h_states[t - 1] if t > 0 else h_states[0]
            c_prev = c_states[t - 1] if t > 0 else c_states[0]
            
            concat_input = np.vstack((h_prev, x_t))
            
            f_gates[t] = sigmoid(np.dot(self.weights_f, concat_input) + self.bias_f)
            i_gates[t] = sigmoid(np.dot(self.weights_i, concat_input) + self.bias_i)
            c_bar_gates[t] = np.tanh(np.dot(self.weights_c, concat_input) + self.bias_c)
            
            c_states[t] = f_gates[t] * c_prev + i_gates[t] * c_bar_gates[t]
            
            o_gates[t] = sigmoid(np.dot(self.weights_o, concat_input) + self.bias_o)
            h_states[t] = o_gates[t] * np.tanh(c_states[t])
            
        cache = (x_sequence, h_states, c_states, f_gates, i_gates, c_bar_gates, o_gates)
        return h_states[:-1], cache

    def backward_pass(self, dh_states, cache):
        x_sequence, h_states, c_states, f_gates, i_gates, c_bar_gates, o_gates = cache
        seq_length = x_sequence.shape[0]
        
        d_weights_f = np.zeros_like(self.weights_f)
        d_weights_i = np.zeros_like(self.weights_i)
        d_weights_c = np.zeros_like(self.weights_c)
        d_weights_o = np.zeros_like(self.weights_o)
        
        d_bias_f = np.zeros_like(self.bias_f)
        d_bias_i = np.zeros_like(self.bias_i)
        d_bias_c = np.zeros_like(self.bias_c)
        d_bias_o = np.zeros_like(self.bias_o)
        
        dh_next = np.zeros((self.hidden_size, 1))
        dc_next = np.zeros((self.hidden_size, 1))
        
        dx_sequence = np.zeros_like(x_sequence)
        
        for t in reversed(range(seq_length)):
            dh_t = dh_states[t].reshape(-1, 1) + dh_next
            dc_t = dh_t * o_gates[t] * tanh_derivative(c_states[t]) + dc_next
            
            do_t = dh_t * np.tanh(c_states[t]) * sigmoid_derivative(o_gates[t])
            dc_bar_t = dc_t * i_gates[t] * tanh_derivative(c_bar_gates[t])
            di_t = dc_t * c_bar_gates[t] * sigmoid_derivative(i_gates[t])
            
            c_prev = c_states[t - 1] if t > 0 else np.zeros_like(c_states[0])
            df_t = dc_t * c_prev * sigmoid_derivative(f_gates[t])
            
            x_t = x_sequence[t].reshape(-1, 1)
            h_prev = h_states[t - 1] if t > 0 else np.zeros_like(h_states[0])
            concat_input = np.vstack((h_prev, x_t))
            
            d_weights_o += np.dot(do_t, concat_input.T)
            d_bias_o += do_t
            
            d_weights_c += np.dot(dc_bar_t, concat_input.T)
            d_bias_c += dc_bar_t
            
            d_weights_i += np.dot(di_t, concat_input.T)
            d_bias_i += di_t
            
            d_weights_f += np.dot(df_t, concat_input.T)
            d_bias_f += df_t
            
            d_concat_input = (np.dot(self.weights_o.T, do_t) + 
                              np.dot(self.weights_c.T, dc_bar_t) + 
                              np.dot(self.weights_i.T, di_t) + 
                              np.dot(self.weights_f.T, df_t))
            
            dh_next = d_concat_input[:self.hidden_size]
            dx_sequence[t] = d_concat_input[self.hidden_size:].reshape(-1)
            dc_next = f_gates[t] * dc_t
            
        gradients = (d_weights_f, d_bias_f, d_weights_i, d_bias_i, 
                     d_weights_c, d_bias_c, d_weights_o, d_bias_o)
                     
        return dx_sequence, gradients

    def update_parameters(self, gradients, learning_rate):
        (d_weights_f, d_bias_f, d_weights_i, d_bias_i, 
         d_weights_c, d_bias_c, d_weights_o, d_bias_o) = gradients
         
        np.clip(d_weights_f, -1, 1, out=d_weights_f)
        np.clip(d_weights_i, -1, 1, out=d_weights_i)
        np.clip(d_weights_c, -1, 1, out=d_weights_c)
        np.clip(d_weights_o, -1, 1, out=d_weights_o)
         
        self.weights_f -= learning_rate * d_weights_f
        self.bias_f -= learning_rate * d_bias_f
        self.weights_i -= learning_rate * d_weights_i
        self.bias_i -= learning_rate * d_bias_i
        self.weights_c -= learning_rate * d_weights_c
        self.bias_c -= learning_rate * d_bias_c
        self.weights_o -= learning_rate * d_weights_o
        self.bias_o -= learning_rate * d_bias_o