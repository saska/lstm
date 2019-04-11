import random

import numpy as np

from functions import Dense, sigmoid, d_sigmoid, tanh, d_tanh, L2_loss

from lstm import LSTM, LSTM_unit


TEST_COUNT = 10 #How many times most tests are ran
def random_params():
    time_steps = np.random.randint(1, 10) * 10
    hidden_dim = np.random.randint(90, 110)
    output_dim = np.random.randint(1,10)
    x_dim = np.random.randint(5, 20)
    n_examples = np.random.randint(120,130)
    batch_size = time_steps // 10
    return (time_steps, hidden_dim, output_dim, 
            x_dim, n_examples, batch_size)

#np.random.seed(1)

def minibatch_gen(data, target, batch_size, shuffle=True):
    if shuffle:
        perm = np.random.permutation(len(target))
        target, data = target[perm], data[perm]
    num_batches = int(np.ceil(len(target) / batch_size))
    for i in range(1,num_batches+1):
        yield data[:,(i-1)*batch_size:i*batch_size,:], \
              target[:,(i-1)*batch_size:i*batch_size,:]

def test_net_forward_prop_dims():
    for i in range(TEST_COUNT):
        (time_steps, hidden_dim, output_dim,
         x_dim, n_examples, batch_size) = random_params()
        arr = np.random.randn(time_steps, n_examples, x_dim)
        assert len(arr) == time_steps
        targets = np.random.randn(time_steps, n_examples, output_dim)
        assert len(targets) == time_steps
        activation = Dense(hidden_dim, output_dim)
        net = LSTM(hidden_dim, x_dim, batch_size=batch_size)
        for data, target in minibatch_gen(arr, targets, batch_size):
            states, caches, preds, losses = net.forward(data, target)
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

def test_lstm_net_forward_backward():
    for i in range(TEST_COUNT):
        (time_steps, hidden_dim, output_dim,
         x_dim, n_examples, batch_size) = random_params()

        arr = np.random.randn(time_steps, n_examples, x_dim)
        assert len(arr) == time_steps
        targets = np.random.randn(time_steps, n_examples, output_dim)
        assert len(targets) == time_steps
        activation = Dense(hidden_dim, output_dim)
        net = LSTM(hidden_dim, x_dim, output_dim=output_dim, batch_size=batch_size)
        for data, target in minibatch_gen(arr, targets, batch_size):
            params = net.cell.params
            net.backward(*(net.forward(data, target)))
            #check the params were updated
            params != net.cell.params
            for k, v in params.items():
                #check param dims are identical to previous iteration
                assert v['w'].shape == net.cell.params[k]['w'].shape
                assert v['b'].shape == net.cell.params[k]['b'].shape

def test_dense_layer_dims():
    for i in range(TEST_COUNT):
        n_layers = np.random.randint(1, 15)
        x_dim = np.random.randint(5, 25)
        n_examples = np.random.randint(100, 110)
        dim_list = [np.random.randint(1, 200) for _ in range(n_examples)]
        layers = []
        layers.append(Dense(x_dim, dim_list[0]))
        for i in range(len(dim_list)-1):
            layers.append(Dense(dim_list[i], dim_list[i+1]))
        x = np.random.randn(n_examples, x_dim)
        a = x.T
        #test forward
        for l in layers:
            a = l.forward(a)
            assert(a.shape == (l.w.shape[0], n_examples))
        #test backward
        da = a.T
        for l in reversed(layers):
            da = l.backward(da)
            assert(da.shape == (n_examples, l.w.shape[1]))

def _cell_forward_calcs():
    """TODO This doesn't do anything yet
    """
    for i in range(TEST_COUNT):
        (time_steps, hidden_dim, output_dim,
         x_dim, n_examples, batch_size) = random_params()

        arr = np.random.randn(n_examples, x_dim)
        cell = LSTM_unit(hidden_dim, x_dim, batch_size=n_examples)
        a_prev = None
        c_prev = None
        for i in range(30):
            state, cache = cell.forward(arr, a_prev, c_prev)
            a_prev = state['a_out']
            c_prev = state['c_out']
            np.assert_almost_equal(cell.state['c'], tanh(np.dot()))

def test_net_forward_calcs():
    pass

def test_grads():
    (time_steps, hidden_dim, output_dim,
     x_dim, n_examples, batch_size) = random_params()
    batch_size = time_steps // 10
    arr = np.random.randn(time_steps, n_examples, x_dim)
    targets = np.random.randn(time_steps, n_examples, output_dim)
    activation = Dense(hidden_dim, output_dim)
    #Set learning rate to 0 to not update any grads
    net = LSTM(hidden_dim, x_dim, batch_size=n_examples, 
               output_dim=output_dim, learning_rate=0)
    delta = 1e-5
    for i in range(TEST_COUNT):
        old_params = net.cell.params
        for _, p in net.cell.params.items():
            p['w'] += delta
            p['b'] += delta
        plus_loss = np.array(net.epoch(arr, targets))
        for _, p in net.cell.params.items():
            p['w'] -= 2 * delta
            p['b'] -= 2 * delta
        minus_loss = np.array(net.epoch(arr, targets))
        print(plus_loss.shape, minus_loss.shape)
        num_grad = (plus_loss - minus_loss) * (0.5 * delta)
        print(num_grad.shape)
        net.cell.learning_rate = delta
        base_loss = np.array(net.epoch(arr, targets))
        updated_loss = np.array(net.epoch(arr, targets))
        assert not np.array_equal(base_loss, updated_loss)
        analytical_grad = updated_loss - base_loss
        tr = abs(num_grad - analytical_grad) < 1e-6
        unique, counts = np.unique(tr, return_counts=True)
        print(dict(zip(unique, counts)))
        assert 0


def test_L2_loss_gradient():
    delta = 1e-5
    loss = L2_loss.loss
    dloss = L2_loss.dloss
    time_steps = np.random.randint(1, 10) * 10
    n_examples = np.random.randint(120,130)
    y_dim = np.random.randint(3,8)
    y_hat = np.random.randn(time_steps, n_examples, y_dim)
    y = np.random.randn(time_steps, n_examples, y_dim)
    num_grads = 1/(2 * delta) * (loss(y_hat + delta, y) - loss(y_hat - delta, y))
    analytical_grads = dloss(y_hat, y)
    np.testing.assert_array_almost_equal(num_grads, analytical_grads, decimal=8)

def test_Dense_gradient():
    for i in range(TEST_COUNT):
        (time_steps, hidden_dim, output_dim,
            x_dim, n_examples, batch_size) = random_params()
        delta = 1e-5
        l = Dense(x_dim, output_dim)

        a = np.random.randn(x_dim, n_examples)
        #Ugliest thing I've ever written, might break with any change
        orig_w = l.w.copy()
        orig_b = l.b.copy()
        num_grad = (l.forward(a - delta) - l.forward(a + delta)) / (2 * delta)
        l.w += delta
        w_grad_tmp = l.forward(a)
        l.w -= 2 * delta
        w_grad = (w_grad_tmp - l.forward(a)) / (2 * delta)
        l.w = orig_w
        l.b += delta
        b_grad_tmp = l.forward(a)
        l.b -= 2 * delta
        b_grad = (b_grad_tmp - l.forward(a)) / (2 * delta)
        l.b = orig_b
        #This needs to be run last since it resets layer cache
        a = l.forward(a)
        da = l.backward(a.T)

        #np.testing.assert_almost_equal(num_grad, da)
        np.testing.assert_almost_equal(w_grad, l.dw)
        np.testing.assert_almost_equal(b_grad, l.db)

