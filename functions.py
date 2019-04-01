import numpy as np

def sigmoid(Z):
    #Also returns original to help with backprop
    return 1/(1+np.exp(-Z)), Z

#NOTE TODO: Careful with the derivative functions here; gates already activated (so e.g state['c'] is already sigmoid(c))
def d_sigmoid(cache):
    s, _ = sigmoid(cache)
    return s * (1-s)

def tanh(Z):
    #Also returns original to help with backprop
    A, _ = sigmoid(Z * 2) 
    return A * 2 - 1, Z
    
def d_tanh(self, dA, cache):
    t, _ = tanh(cache)
    dZ = dA * (1 - t**2)
    assert (dZ.shape == cache.shape)
    return dZ
    
class L2_loss:
    @classmethod
    def loss(self, y_hat, y):
        return (y_hat - y)**2

    @classmethod
    def dloss(self, y_hat, y):
        print(len(y_hat), len(y))
        print(y_hat)
        #print(y_hat[0], y[0])
        return (y_hat - y) * 2

class Unit_activation:
    @classmethod
    def activation(self, Z):
        return Z

    @classmethod
    def dactivation(self, Z):
        return np.ones_like(Z)

class Dense:
    def __init__(self, input_dim, output_dim, activation=tanh, learning_rate=1e-3):
        self.w = xavier_init((input_dim, output_dim))
        self.activation = activation
    
    def activate(self, Z):
        return self.activation(np.dot(Z, self.w))

    def dactivate()
    

#class Sigmoid_activation:
#    @classmethod
#    def activation(self, Z)

#def xavier_init(*args):
#    return np.random.randn(*args) * np.sqrt(2 / sum(*args))
xavier_init = lambda dims: np.random.randn(*dims) * np.sqrt(2 / (sum(dims)))