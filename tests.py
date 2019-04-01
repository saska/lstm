import numpy as np
from lstm import LSTM, LSTM_unit
#np.random.seed(1)

def minibatch_gen(data, target, batch_size, shuffle=True):
    
    if shuffle:
        perm = np.random.permutation(len(target))
        target, data = target[perm], data[perm]
        
    num_batches = int(np.ceil(len(target) / batch_size))
    
    for i in range(1,num_batches+1):
        yield data[:,(i-1)*batch_size:i*batch_size,:], \
              target[:,(i-1)*batch_size:i*batch_size]

def test_model_forward_prop():
    for i in range(10):
        time_steps = np.random.randint(1, 10) * 10
        hidden_dim = np.random.randint(90, 110)
        x_dim = np.random.randint(5, 20)
        n_examples = np.random.randint(120,130)
        batch_size = time_steps // 10
        arr = np.random.randn(time_steps, n_examples, x_dim)
        assert len(arr) == time_steps
        targets = np.random.randn(time_steps, n_examples)
        assert len(targets) == time_steps
        net = LSTM(hidden_dim, x_dim, batch_size=batch_size)
        for data, target in minibatch_gen(arr, targets, batch_size):
            states, caches, preds, losses = net.model_forward(data, target)
            assert len(states) == time_steps
            assert len(caches) == time_steps
            assert len(preds) == time_steps
            assert len(losses) == time_steps
            for state, cache, pred, loss in zip(states, caches, preds, losses):
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
               
def test_model_forward_backward():
    for i in range(10):
        time_steps = np.random.randint(1, 10) * 10
        hidden_dim = np.random.randint(90, 110)
        x_dim = np.random.randint(5, 20)
        n_examples = np.random.randint(120,130)
        batch_size = time_steps // 10
        arr = np.random.randn(time_steps, n_examples, x_dim)
        assert len(arr) == time_steps
        targets = np.random.randn(time_steps, n_examples)
        assert len(targets) == time_steps
        net = LSTM(hidden_dim, x_dim, batch_size=batch_size)
        for data, target in minibatch_gen(arr, targets, batch_size):
            net.model_backward(*(net.model_forward(data, target)))