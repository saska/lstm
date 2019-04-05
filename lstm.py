import numpy as np

from functions import (L2_loss, Unit_activation, d_sigmoid, d_tanh, sigmoid,
                       tanh, xavier_init, Dense)

class Model:
    def __init__(layers, loss):
        self.layers = layers
        self.loss = loss

    def forward(data, targets):
        for l in self.layers:
            pass

    def backward():
        for l in reversed(self.layers):
            pass

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
    def __init__(self, hidden_dim, x_dim, batch_size=1, learning_rate=1e-4, output_dim=3, 
                       loss=L2_loss, activation=Dense, init=xavier_init, peephole=True):
        self.cell = LSTM_unit(hidden_dim, x_dim, batch_size=batch_size, init=init, peephole=peephole)
        self.learning_rate = learning_rate
        self.loss = loss.loss
        self.dloss = loss.dloss
        self.batch_size = batch_size
        self.activation = activation(hidden_dim, output_dim)

    def forward(self, inputs, targets, a_prev=None, c_prev=None):
        states, caches, preds, losses = [], [], [], []
        inputs = [inputs[t,:,:] for t in range(inputs.shape[0])]
        targets = [targets[t,:].reshape(1, targets.shape[1]) for t in range(targets.shape[0])]
        for x, y in zip(inputs, targets):
            state, cache = self.cell.forward(x, a_prev, c_prev)
            states.append(state)
            caches.append(cache)
            pred = self.activation.forward(state['a_out'])
            preds.append(pred)
            losses.append(self.loss(pred, y))
            a_prev, c_prev = state['a_out'], state['c_out']
        return states, caches, np.array(preds), np.array(targets)

    def backward(self, states, caches, preds, targets):
        assert len(states) == len(caches) == len(preds)
        d_loss = np.mean(self.dloss(preds, targets), axis=0).T
        da_next = self.activation.backward(d_loss).T
        dc_next = np.zeros_like((states[0]['c_out']))
        grads = {k: np.zeros((self.cell.hidden_dim, 1)) for k in ['c', 'u', 'o', 'f']}
        for t in reversed(range(len(states))):
            self.cell.state = states[t]
            da_next, dc_next, grad_adds = self.cell.backward(da_next, dc_next)
            for gate in ['c', 'u', 'o', 'f']:
                grads[gate] += np.mean(grad_adds[gate], axis=1, keepdims=True)
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
            k: {'w': self.init((self.hidden_dim, self.concat_dim)), 'b': np.zeros((self.hidden_dim, 1))} for k in ['c', 'u', 'o', 'f']
        }

    def forward(self, x, a_prev, c_prev):
        a_prev = a_prev if a_prev is not None else np.zeros((self.hidden_dim, self.batch_size))
        c_prev = c_prev if c_prev is not None else np.zeros((self.hidden_dim, self.batch_size))
        self.state['c_in'] = c_prev
        x = x.reshape((self.x_dim, self.batch_size))
        self.z = np.vstack((x, a_prev)).reshape((self.concat_dim, self.batch_size))
        cache = {}
        for k, func in self.funcs.items():
            self.state[k], cache[k] = func['a'](np.dot(self.params[k]['w'], self.z) + self.params[k]['b'])
        self.state['c_out'] = self.state['f'] * c_prev + self.state['u'] * self.state['c']
        self.state['a_out'] = self.state['o'] * tanh(self.state['c_out'])[0]
        return self.state, cache

    def backward(self, da_next, dc_next):
        #TODO confirm this is the right math
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
            print(self.params[gate]['w'].shape, grads[gate].shape, self.z.shape)
            grads[gate] = np.clip(grads[gate], -self.grad_clip, self.grad_clip)
            #TODO check this math (the np.mean is maybe dumb and breaks the whole thing?)
            self.params[gate]['w'] += np.outer(grads[gate], np.mean(self.z.T, axis=0, keepdims=True))
            self.params[gate]['b'] += grads[gate]