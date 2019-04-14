import numpy as np
import time
from lstm import LSTM


def minibatch_gen(data, target, batch_size, shuffle=True):
    if shuffle:
        perm = np.random.permutation(target.shape[1])
        target, data = target[:,perm,:], data[:,perm,:]
    num_batches = int(np.ceil(target.shape[1] / batch_size))
    for i in range(1,num_batches+1):
        yield data[:,(i-1)*batch_size:i*batch_size,:], \
              target[:,(i-1)*batch_size:i*batch_size,:]
time_steps = 10
x_dim = 5
hidden_dim = 100
output_dim = 8
n_examples = 100
batch_size = 100
x = np.random.randn(time_steps, n_examples, x_dim)
y = np.random.randn(time_steps, n_examples, output_dim)
net = LSTM(hidden_dim, x_dim, output_dim=output_dim, batch_size=batch_size, learning_rate=1e-5)
for i in range(10000):
    start = time.time()
    loss = 0
    for data, targets in minibatch_gen(x, y, batch_size):
        loss += np.mean(net.epoch(data, targets))
    print('Epoch {}: loss: {} time: {}'.format(i, loss, time.time() - start), end='\r', flush=True)



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
