import numpy as np

from functions import sigmoid, d_sigmoid, tanh, d_tanh, L2_loss, xavier_init, Unit_activation

class LSTM:
    """Class for LSTM network.
    
    Inputs:
        hidden_dim: size of the hidden layer, passed to the LSTM cell
        x_dim: size of network inputs (data passed to the network)
        loss: loss class, consisting of a loss function (loss) and it's derivative 
            w.r.t. prediction (dloss) as class methods. For an example, see functions.py
        init: initialization function, should take two dimensions x and y (integers) 
            as input and return an array of shape x, y. Passed to the LSTM cell.
        peephole: whether to use a peephole connection, boolean. Passed to the LSTM cell.
    """
    def __init__(self, hidden_dim, x_dim, batch_size=1, learning_rate=1e-3, loss=L2_loss, activation=Unit_activation, init=xavier_init, peephole=True):
        self.cell = LSTM_unit(hidden_dim, x_dim, batch_size=batch_size, init=init, peephole=peephole)
        self.learning_rate = learning_rate
        self.loss = loss.loss
        self.dloss = loss.dloss
        self.batch_size = batch_size
        self.activation = activation.activation
        self.dactivation = activation.dactivation

    def model_forward(self, inputs, targets, a_prev=None, c_prev=None):
        state, cache = self.cell.forward(inputs, a_prev, c_prev)
        pred = self.activation(state['a_out'])
        loss = self.loss(pred, targets)
        a_prev, c_prev = state['a_out'], state['c_out']
        return state, cache, pred, loss

    def model_backward(self, states, caches, preds, targets):
        assert len(states) == len(caches) == len(preds)
        d_loss = self.dloss(preds, targets)
        dact = self.dactivation(states[len(states) - 1]['a_out'])
        da_next = d_loss * dact
        dc_next = np.zeros_like((states[0]['dc_out']))
        grads = {k: np.zeros((self.cell.hidden_dim, 1)) for k in ['c', 'u', 'o', 'f']}
        for t in reversed(range(len(states))):
            self.cell.state = states[t]
            da_next, dc_next, grad_adds = self.cell.backward(da_next, dc_next)
            for gate in ['c', 'u', 'o', 'f']:
                grads[gate] += grad_adds[gate]
        self.cell.update_params(grads)
        
class LSTM_unit:
    """Class for LSTM cell.

    Inputs:
        hidden_dim: Size of the hidden layer
        x_dim: Size of network inputs
        init: Initialization function, should take two dimensions x and y (integers) 
            as input and return an array of shape x, y
        grad_clip: Gradients will be clipped between -value and value. Pass None to disable clipping 
        peephole: Whether to use a peephole connection, boolean
    """
    def __init__(self, hidden_dim, x_dim, batch_size=1, init=xavier_init, grad_clip=1, peephole=True):
        #TODO figure out peephole again or scrap it
        self.hidden_dim = hidden_dim
        self.x_dim = x_dim
        self.concat_dim = hidden_dim + x_dim
        self.batch_size = batch_size
        self.init = init
        self.init_state()
        self.init_params()
        self.grad_clip = grad_clip
        self.peephole = peephole
        self.funcs = {
            'c': {'a': tanh, 'd': d_tanh},
            'u': {'a': sigmoid, 'd': d_sigmoid},
            'o': {'a': sigmoid, 'd': d_sigmoid},
            'f': {'a': sigmoid, 'd': d_sigmoid},
        }

    def init_state(self):
        self.state = {k: np.zeros((self.hidden_dim, self.batch_size)) for k in ['c', 'u', 'o', 'f']}

    def init_params(self):
        self.params = {
            k: {'w': self.init(self.hidden_dim, self.concat_dim), 'b': np.zeros((self.hidden_dim, 1))} for k in ['c', 'u', 'o', 'f']
        }

    def forward(self, x, a_prev, c_prev):
        a_prev = a_prev if a_prev is not None else np.zeros((self.batch_size, self.hidden_dim))
        c_prev = c_prev if c_prev is not None else np.zeros((self.batch_size, self.hidden_dim))
        x = x.reshape((self.batch_size, self.x_dim))
        self.z = np.hstack((x, a_prev)).reshape((self.concat_dim, self.batch_size))
        cache = {}
        for k, func in self.funcs.items():
            self.state[k], cache[k] = func['a'](np.dot(self.params[k]['w'], self.z) + self.params[k]['b'])
        self.state['c_out'] = self.state['f'] * c_prev.T + self.state['u'] * self.state['c']
        self.state['a_out'] = self.state['o'] * tanh(self.state['c_out'])[0]
        return self.state, cache


    def backward(self, da_next, dc_next):
        dc_out = self.state['o'] * da_next + dc_next
        diff = lambda gate: self.funcs[gate]['d'](self.state[gate])
        grads = {}
        grads['c'] = diff('c') * self.state['u'] * dc_out
        grads['u'] = diff('u') * self.state['c_out'] * dc_out
        grads['o'] = diff('o') * self.state['c'] * da_next
        grads['f'] = diff('f') * self.state['c_in'] * dc_out
        dz = np.zeros_like(self.z)
        for gate in ['c', 'u', 'o', 'f']:
            dz += np.dot(self.params[gate]['w'].T, grads[gate])
        da_in = dz[self.x_dim:]
        dc_in = dc_out * self.state['f']
        return da_in, dc_in, grads


    def update_params(self, grads):
        for gate in ['c', 'u', 'o', 'f']:
            grads[gate] = np.clip(grads[gate], -self.grad_clip, self.grad_clip)
            self.params[gate]['w'] += np.outer(grads[gate], self.z)
            self.params[gate]['b'] += grads[gate]
        

