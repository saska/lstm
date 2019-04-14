import numpy as np

#TODO: Figure out wrapping and maybe superclassing all activations? Might work neat
def sigmoid(Z):
    #Also returns original to help with backprop
    return 1/(1+np.exp(-Z)), Z

def d_sigmoid(cache):
    s, _ = sigmoid(cache)
    return s * (1-s)

def tanh(Z):
    #Also returns original to help with backprop
    A, _ = sigmoid(Z * 2) 
    return A * 2 - 1, Z
    
def d_tanh(cache):
    t, _ = tanh(cache)
    return (1 - t**2)
    
class L2_loss:
    @classmethod
    def loss(self, y_hat, y):
        return (y_hat - y) ** 2

    @classmethod
    def dloss(self, y_hat, y):
        return (y_hat - y) * 2

class Dense:
    """Class for dense (fully connected) layer.
    
    """
    def __init__(self, input_dim, output_dim, activation=tanh, dactivation=d_tanh, learning_rate=1e-3):
        self.w = xavier_init((output_dim, input_dim))
        self.b = np.zeros((output_dim, 1))
        self.dw = np.zeros_like(self.w)
        self.db = np.zeros_like(self.b)
        self.activation = activation
        self.dactivation = dactivation
        self.learning_rate = learning_rate

    def forward(self, a_prev):
        self.a_prev = a_prev
        A, self.cache = self.activation(np.dot(self.w, a_prev) + self.b)
        return A

    def backward(self, da, cache=None):
        cache = self.cache if cache is None else cache
        dZ = da * self.dactivation(cache.T)
        self.dw += 1/self.a_prev.shape[0] * np.dot(self.a_prev, dZ).T
        self.db += 1/self.a_prev.shape[0] * np.sum(dZ, keepdims=True)
        return np.dot(dZ, self.w)

    def update_params(self):
        self.w -= self.dw * self.learning_rate
        self.b -= self.db * self.learning_rate
 
def xavier_init(dims):
    """Xavier initialization.
    Input: n-tuple of dimensions
    Returns: Xavier-initialized array of shape dims
    """
    return np.random.randn(*dims) * np.sqrt(2 / sum(dims))
