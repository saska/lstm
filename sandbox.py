import numpy as np
import time
from lstm import LSTM
from matplotlib import pyplot as plt

def minibatch_gen(data, target, batch_size, shuffle=True):
    if shuffle:
        perm = np.random.permutation(target.shape[1])
        target, data = target[:,perm,:], data[:,perm,:]
    num_batches = int(np.ceil(target.shape[1] / batch_size))
    for i in range(1,num_batches+1):
        yield data[:,(i-1)*batch_size:i*batch_size,:], \
              target[:,(i-1)*batch_size:i*batch_size,:]

def simplefunc():
    time_steps = 10
    x_dim = 8
    hidden_dim = 8
    output_dim = 8
    n_examples = 2048
    batch_size = 256
    x = np.random.randn(time_steps, n_examples, x_dim)
    y = np.random.randn(time_steps, n_examples, output_dim)
    x = np.ones((time_steps, n_examples, x_dim))
    y = np.ones((time_steps, n_examples, output_dim)) / 4.5
    y[4:,:,:] = y[4:,:,:] * 3.6
    net = LSTM(hidden_dim, x_dim, output_dim=output_dim, learning_rate=1e-5)
    losses = []
    for i in range(5000):
        start = time.time()
        loss = 0
        for data, targets in minibatch_gen(x, y, batch_size):
            loss += np.mean(net.epoch(data, targets))
        losses.append(loss)
        print('Epoch {}: loss: {} time: {}'.format(i, loss, time.time() - start), end='\r', flush=True)
    
    print('\nEpoch {}: loss: {} time: {}'.format(i, loss, time.time() - start), end='\r', flush=True)
    plt.plot(losses)
    plt.show()

def courseratest():
    np.random.seed(1)
    x_dim = 3
    n_examples = 10
    time_steps = 7
    hidden_dim = 5
    
    da = np.random.randn(5, 10, 4)
    x = np.ones((time_steps, n_examples, x_dim))
    for i in range(time_steps):
        for j in range(n_examples):
            for k in range(x_dim):
                x[i,j,k] = np.random.randn()
    from functions import xavier_init
    net = LSTM(hidden_dim, x_dim)
    states, caches, preds, ys = net.forward(x, np.zeros((time_steps, n_examples, 1)))
    # print(states[-1]['z'])
    # print(states[-1]['c_out'])
    # print(states[-1]['f'])
    # print(states[-1]['u'])
    # print(states[-1]['o'])
    da_next = np.zeros_like(da[:,:,0])
    dc_next = np.zeros_like(states[0]['c'])
    grads = net.cell.init_grads()
    for t in reversed(range(4)):
        da_next, dc_next, grad_adds = net.cell.backward(states[t], da[:,:,t] + da_next, dc_next)
        for gate in ['c', 'u', 'o', 'f']:
            grads[gate]['w'] += grad_adds[gate]['w']
            grads[gate]['b'] += grad_adds[gate]['b']
        print(grad_adds['f']['b'])
if __name__ == "__main__":
    simplefunc()

# class Model:
#     def __init__(layers, loss):
#         self.layers = layers
#         self.loss = loss

#     def forward(data, targets):
#         for l in self.layers:
#             data = l.forward(data)
#         return data

#     def backward(grads):
#         for l in reversed(self.layers):
#             grads = l.backward(grads)

#     def loss(data, targets):
#         return self.loss.loss(data, targets)

#     def dloss(data, targets):
#         return self.loss.dloss(data, targets)

# class Default_LSTM(Model):
#     def __init__(lstm_hidden_dim, x_dim, batch_size=1, learning_rate=1e-4, output_dim=1,
#                  loss=L2_loss, activation=Dense, init=xavier_init, peephole=False):
#         self.lstm = LSTM(hidden_dim, x_dim, batch_size=batch_size, 
#                          learning_rate=learning_rate, output_dim=output_dim, 
#                          loss=loss, activation=activation, init=init, peephole=peephole)
#         self.activation_layer = activation(lstm_hidden_dim, output_dim)
#         super.__init__([self.lstm, self.activation_layer], loss)
