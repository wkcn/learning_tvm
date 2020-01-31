# reference: https://github.com/apache/incubator-tvm/tree/master/topi/python/topi/nn/softmax.py
import tvm
import mxnet as mx
import numpy as np

TVM_FUNC = dict()
def tvm_func(func):
    def wrapper_func(*args, **kwargs):
        if func not in TVM_FUNC:
            # register
            tvm_args = [None] * len(args)
            tvm_placeholders = []
            for i, a in enumerate(args):
                if isinstance(a, tvm.nd.NDArray):
                    a = tvm.placeholder(a.shape)
                    tvm_placeholders.append(a)
                tvm_args[i] = a
            tvm_out = func(*tvm_args, **kwargs)
            out_shape = tvm_out.shape
            s = tvm.create_schedule(tvm_out.op)
            mod = tvm.build(s, tuple(tvm_placeholders) + (tvm_out,))
            TVM_FUNC[func] = (mod, out_shape)
        mod, out_shape = TVM_FUNC[func]
        out = tvm.nd.empty(out_shape)
        mod(*(args + (out,)))
        return out
    return wrapper_func

@tvm_func
def softmax(x, axis=-1):
    shape = x.shape
    if axis < 0:
        axis = len(shape) + axis
    assert 0 <= axis < len(shape)

    def insert_reduce_axis(indices, k):
        return indices[:axis] + (k, ) + indices[axis:]

    def get_non_reduce_indices(indices):
        return tuple([var for (i, var) in enumerate(indices) if i != axis])

    def _compute_max(*indices):
        k = tvm.reduce_axis((0, shape[axis]), name='k')
        eval_range = insert_reduce_axis(indices, k)
        return tvm.max(x[eval_range], axis=k)

    def _compute_exp(*indices):
        non_reduce_indices = get_non_reduce_indices(indices)
        return tvm.exp(x[indices] - max_elem[non_reduce_indices])

    def _compute_sum(*indices):
        k = tvm.reduce_axis((0, shape[axis]), name='k')
        eval_range = insert_reduce_axis(indices, k)
        return tvm.sum(exp[eval_range], axis=k)

    def _normalize(*indices):
        non_reduce_indices = get_non_reduce_indices(indices)
        return exp[indices] / expsum[non_reduce_indices]

    reduced_shape = get_non_reduce_indices(shape)
    max_elem = tvm.compute(reduced_shape, _compute_max, name='max_elem')
    exp = tvm.compute(shape, _compute_exp, name='exp')
    expsum = tvm.compute(reduced_shape, _compute_sum, name='expsum')
    return tvm.compute(shape, _normalize, name='norm')


dtype = 'float32'
N = 2
C = 3
x = np.random.normal(size=(N, C)).astype(dtype)
print(x)

tvm_x = tvm.nd.array(x)
tvm_out = softmax(tvm_x, axis=-1)
print(tvm_out)

mx_x = mx.nd.array(x)
mx_out = mx.nd.softmax(mx_x, axis=-1)
print(mx_out)
