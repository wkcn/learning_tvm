import tvm
import topi
import numpy as np
import matplotlib.pyplot as plt
import time


def get_gaussian_map_sum(shape, data):
    '''
    Inputs:
        shape: tuple
            the shape of output
        data: np.ndarray
            [[x, y, sigma], ...]
    Return:
        np.ndarray (shape[0], shape[1])
    '''
    height, width = shape
    centers = data[:, :2]
    sigmas = data[:, 2]
    center_xs = centers[:, 0].reshape((-1, 1))
    center_ys = centers[:, 1].reshape((-1, 1))

    dxs = np.arange(width).reshape((1, -1)) - center_xs  # (n, width)
    dys = np.arange(height).reshape((1, -1)) - center_ys  # (n, height)

    sigmas2 = np.square(sigmas).reshape((-1, 1, 1))

    # (n, height, width)
    ds2 = np.square(dxs).reshape((-1, 1, width)) + \
        np.square(dys).reshape((-1, height, 1))

    # Optimization
    G = np.exp(-ds2 / (2.0 * sigmas2)) / (2.0 * np.pi * sigmas2)
    return G.sum(0)

def _get_gaussian_map_sum_tvm_mod():
    rows, cols = tvm.var('rows'), tvm.var('cols') # the shape of output
    n = tvm.var('n') # the number of samples
    data = tvm.placeholder((n, 3), name='data')
    ni = tvm.reduce_axis((0, n), name='ni')
    pi = tvm.const(np.pi)
    def _gaussian_map_sum(i, j):
        # i is row, j is col
        x, y = data[ni, 0], data[ni, 1]
        sigma = data[ni, 2]
        sigma2 = sigma * sigma
        v = tvm.if_then_else(tvm.all(x >= 0, x < cols, y >= 0, y < rows),
            tvm.exp(-(topi.power((x - j), 2) + topi.power((y - i), 2)) / (2 * sigma2)) / (2 * pi * sigma2),
            0
        )
        return tvm.sum(v, axis=ni)
    out = tvm.compute((rows, cols), _gaussian_map_sum, name='out')
    s = tvm.create_schedule(out.op)
    out_i = s[out].fuse(*out.op.axis)
    s[out].parallel(out_i)
    print(tvm.lower(s, [data], simple_mode=True))
    return tvm.build(s, [data, out])

_gaussian_map_sum_tvm_mod = _get_gaussian_map_sum_tvm_mod()
def get_gaussian_map_sum_tvm(shape, data):
    tvm_out = tvm.nd.empty(shape)
    tvm_data = tvm.nd.array(data)
    _gaussian_map_sum_tvm_mod(tvm_data, tvm_out)
    return tvm_out.asnumpy()

def run_func(func, times):
    tic = time.time()
    for _ in range(times):
        func()
    return time.time() - tic

def bench_func(func):
    # warm up
    func()
    # test
    t = run_func(func, 1)
    if t < 1:
        times = int(np.ceil(1.0 / t))
        t = run_func(func, times) / times
    return t

N = 1000
S = 30
shape = (S, S)
dtype = 'float32'
centers = np.random.randint(0, S, size=(N, 2)).astype(dtype)
sigmas = np.random.uniform(0, 5, size=(N, 1)).astype(dtype)
data = np.concatenate([centers, sigmas], axis=1)
print(data.shape, data.dtype)

np_out = get_gaussian_map_sum(shape, data)
tvm_out = get_gaussian_map_sum_tvm(shape, data)

np_time = bench_func(lambda : get_gaussian_map_sum(shape, data))
tvm_time = bench_func(lambda : get_gaussian_map_sum_tvm(shape, data))
print("NP: {}, TVM: {}".format(np_time, tvm_time))

plt.subplot(121)
plt.imshow(np_out)
plt.subplot(122)
plt.imshow(tvm_out)
plt.show()

np.testing.assert_allclose(np_out, tvm_out, atol=1e-5)
