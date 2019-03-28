import numpy as np
from lstm import LSTM, LSTM_unit
#np.random.seed(1)

def minibatch_gen(data, target, batch_size, shuffle=True):
    
    if shuffle:
        perm = np.random.permutation(len(target))
        target, data = target[perm], data[perm]
        
    num_batches = int(np.ceil(len(target) / batch_size))
    
    for i in range(1,num_batches+1):
        yield data[(i-1)*batch_size:i*batch_size, :], \
              target[(i-1)*batch_size:i*batch_size]

def test_model_forward_prop():
    for i in range(10):
        time_steps = np.random.randint(1, 10) * 10
        hidden_dim = np.random.randint(90, 110)
        x_dim = np.random.randint(5, 20)
        batch_size = time_steps // 10
        arr = np.random.randn(time_steps, x_dim)
        targets = np.random.randn(time_steps)
        net = LSTM(hidden_dim, x_dim, batch_size=batch_size)
        for data, target in minibatch_gen(arr, targets, batch_size):
            state, cache, pred, loss = net.model_forward(data, target)    
            assert state['c_out'].shape == (hidden_dim, batch_size)
            assert state['a_out'].shape == (hidden_dim, batch_size)
            assert state['c'].shape == (hidden_dim, batch_size)
            assert state['u'].shape == (hidden_dim, batch_size)
            assert state['o'].shape == (hidden_dim, batch_size)
            assert state['f'].shape == (hidden_dim, batch_size)
            assert cache['c'].shape == (hidden_dim, batch_size)
            assert cache['u'].shape == (hidden_dim, batch_size)
            assert cache['o'].shape == (hidden_dim, batch_size)
            assert cache['f'].shape == (hidden_dim, batch_size)
        #assert states['c_out'].shape == 