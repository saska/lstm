import numpy as np

def sigmoid(Z):
    #Also returns original to help with backprop
    return 1/(1+np.exp(-Z)), Z

def d_sigmoid(dA, cache):
    s, _ = sigmoid(cache)
    dZ = dA * s * (1-s)
    assert (dZ.shape == cache.shape)
    return dZ

def tanh(Z):
    #Also returns original to help with backprop
    A, _ = sigmoid(Z * 2)
    return A - 1, Z
    
def d_tanh(dA, cache):
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

#def xavier_init(*args):
#    return np.random.randn(*args) * np.sqrt(2 / sum(*args))
xavier_init = lambda dim1, dim2: np.random.randn(dim1, dim2) * np.sqrt(2 / (dim1 + dim2))