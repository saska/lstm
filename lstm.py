import numpy as np

class L2_loss:

    @classmethod
    def loss(self, y, y_hat):
        return (y_hat - y)**2

    @classmethod
    def dloss(self, y_hat, y):
        return (y_hat, y) * 2

class LSTM:
    pass

class LSTM_unit:
    def __init__(self, hidden_dim, x_dim, loss=L2_loss):
        self.hidden_dim = hidden_dim
        self.x_dim = x_dim

        self.init_state()
        self.init_params()
        self.init_grads()
        self.loss = loss.loss
        self.dloss = loss.dloss

        self.funcs = {
            'c': self.tanh,
            'u': self.sigmoid,
            'o': self.sigmoid,
            'f': self.sigmoid,
        }

    def init_state(self):
        self.state = {k: np.zeros((self.hidden_dim, 1)) for k in ['c', 'u', 'o', 'f']}

    def init_params(self):
        concat_dim = self.hidden_dim + self.x_dim
        xavier_init = lambda dim1, dim2: np.random.randn(dim1, dim2) * np.sqrt(2 / (dim1 + dim2))
        self.params = {
            'wc': xavier_init(self.hidden_dim, concat_dim),
            'wu': xavier_init(self.hidden_dim, concat_dim),
            'wo': xavier_init(self.hidden_dim, concat_dim),
            'wf': xavier_init(self.hidden_dim, concat_dim),
            'bc': np.zeros((self.hidden_dim, 1)),
            'bu': np.zeros((self.hidden_dim, 1)),
            'bo': np.zeros((self.hidden_dim, 1)),
            'bf': np.zeros((self.hidden_dim, 1)),
        }

    def init_grads(self):
        self.grads = {k: np.zeros_like(v) for k, v in self.params.items()}

    def forward(self, a_prev, c_prev, x):
        self.z = np.hstack((x, a_prev))
        cache = {'c_out': c_prev, 'a_out': a_prev}
        for k, v in self.funcs.items():
            self.state[k], cache[k] = v(np.dot(self.params['w'.join(k)], self.z) + self.params['b'.join(k)])
        self.state['c_out'] = np.dot(self.state['f'], c_prev) + np.dot(self.state['i'], self.state['c'])
        self.state['a_out'] = np.dot(self.state['o'], self.tanh(c))
        return self.state, cache

    def backward(self, da, dc):
        dc = self.state['o'] * da + dc #Note: remove dc to remove peephole
        do = self.state['c'] * da
        du = self.state['']



    @staticmethod
    def sigmoid(self, Z):
        #Also returns original to help with backprop
        return 1/(1+np.exp(-Z)), Z

    @staticmethod
    def d_sigmoid(self, dA, cache):
        s, _ = self.sigmoid(cache)
        dZ = dA * s * (1-s)
        assert (dZ.shape == cache.shape)
        return dZ

    @staticmethod
    def tanh(self, Z):
        #Also returns original to help with backprop
        A, _ = self.sigmoid(Z * 2)
        return A - 1, Z

    @staticmethod
    def d_tanh(self, dA, cache):
        t, _ = self.tanh(cache)
        dZ = dA * (1 - t**2)
        assert (dZ.shape == cache.shape)
        return dZ
    