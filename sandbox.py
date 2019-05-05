import pickle
import time
from functools import reduce
import numpy as np
import pandas as pd
#from matplotlib import pyplot as plt
import os

#import quandl
from lstm import LSTM
apipath = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "quandl", "apikey.txt"))
with open(apipath, 'r') as f:
    apikey = f.read()

#quandl.ApiConfig.api_key = apikey

def minibatch_gen(data, target, batch_size, shuffle=True):
    if shuffle:
        perm = np.random.permutation(target.shape[1])
        target, data = target[:,perm,:], data[:,perm,:]
    num_batches = int(np.ceil(target.shape[1] / batch_size))
    for i in range(1,num_batches+1):
        yield data[:,(i-1)*batch_size:i*batch_size,:], \
              target[:,(i-1)*batch_size:i*batch_size,:]

def random_time_batch(df, time_steps=300, batch_size=350):
    batch = []
    for i in range(batch_size):
        start = np.random.randint(len(df) - time_steps)
        batch.append(np.array(df[start:start+time_steps]))
    return np.array(batch).reshape(time_steps, batch_size, len(df.iloc[0]))

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

def quandltest():
    LOAD = False
    
    tickers = ['INTC', 'AMD']
    date = {'gte': '2016-10-10', 'lte': '2017-09-01'}
    columns = {'columns': ['ticker', 'date', 'close']}
    datasets = []
    for ticker in tickers:
        datasets.append(
            quandl.get_table('WIKI/PRICES', qopts=columns, ticker=ticker, date=date)
        )
    for dataset in datasets:
        dataset.rename(columns={'close': dataset['ticker'].iloc[0]}, inplace=True)
        dataset.drop('ticker', axis=1, inplace=True)
        
    df = reduce(lambda l, r: pd.merge(l, r, on='date'), datasets)
    df.index = df['date']
    df.drop('date', axis=1, inplace=True)
    df = (df / df.iloc[-1]).diff()[1:]
    x_dim, output_dim = len(df.iloc[0]), len(df.iloc[0])
            
    hidden_dim = 80
    time_steps = 30
    batch_size = 20
    if LOAD:
        with open('amd_intel_net.pkl', 'rb') as f:
            net = pickle.load(f)
    else:
        net = LSTM(hidden_dim, x_dim, output_dim=output_dim, learning_rate=2e-2)

    for i in range(300000):
            
        batch = np.array(random_time_batch(df, time_steps=time_steps+1, batch_size=batch_size))
        start = time.time()
        x = batch[:-1,:,:]
        y = batch[1:,:,:]
        loss = np.sum(net.epoch(x, y))
        print('Epoch {}: loss: {} time: {}'.format(i, loss, time.time() - start), end='\r', flush=True)
        if i != 0 and i % 5000 == 0:
            with open('amd_intel_net.pkl', 'wb') as f:
                pickle.dump(net, f)
        if i % 3000 == 0:
            net.cell.learning_rate = net.cell.learning_rate * 0.5
            net.activation.learning_rate = net.activation.learning_rate * 0.5

def quandl_nat_gas_vs_crude():
    date = {'gte': '2015-10-10', 'lte': '2017-10-10'}
    df = quandl.get_table('WIKI/PRICES', date=date)
    
    df = reduce(lambda l, r: pd.merge(l, r, on='date'), datasets)
    df.index = df['date']
    df.drop('date', axis=1, inplace=True)
    df = (df / df.iloc[-1]).diff()[1:]
    x_dim, output_dim = len(df.iloc[0]), len(df.iloc[0])
                 
    hidden_dim = 80
    time_steps = 50
    batch_size = 20
    if LOAD:
        with open('crude_vs_nat.pkl', 'rb') as f:
            net = pickle.load(f)
    else:
        net = LSTM(hidden_dim, x_dim, output_dim=output_dim, learning_rate=2e-2)

    for i in range(300000):
            
        batch = np.array(random_time_batch(df, time_steps=time_steps+1, batch_size=batch_size))
        start = time.time()
        x = batch[:-1,:,:]
        y = batch[1:,:,:]
        loss = np.sum(net.epoch(x, y))
        print('Epoch {}: loss: {} time: {}'.format(i, loss, time.time() - start), end='\r', flush=True)
        if i != 0 and i % 5000 == 0:
            with open('crude_vs_nat.pkl', 'wb') as f:
                pickle.dump(net, f)
        if i % 3000 == 0:
            net.cell.learning_rate = net.cell.learning_rate * 0.5
            net.activation.learning_rate = net.activation.learning_rate * 0.5


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
