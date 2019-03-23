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
    A, _ = sigmoid(Z * 2) * 2
    return A - 1, Z
    
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
        return (y_hat - y) * 2

class Unit_activation:
    @classmethod
    def activation(self, Z):
        return Z

    def dactivation(self, Z):
        return np.ones_like(Z)

#def xavier_init(*args):
#    return np.random.randn(*args) * np.sqrt(2 / sum(*args))
xavier_init = lambda dim1, dim2: np.random.randn(dim1, dim2) * np.sqrt(2 / (dim1 + dim2))