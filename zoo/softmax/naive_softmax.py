import tvm
import mxnet as mx
import numpy as np

def get_tvm_softmax2d():
    m, n = tvm.var('m'), tvm.var('n')
    A = tvm.placeholder((m, n), name='a')
    j = tvm.reduce_axis((0, n), name='j')
    A_max = tvm.compute((m, ), lambda i: tvm.max(A[i, j], axis=j), name='A_max')
    # A - A_max
    As = tvm.compute((m, n), lambda i, j: A[i, j] - A_max[i], name='As')
    Aexp = tvm.compute((m, n), lambda i, j: tvm.exp(As[i, j]), name='Aexp')
    j = tvm.reduce_axis((0, n), name='j')
    Aexp_sum = tvm.compute((m, ), lambda i: tvm.sum(Aexp[i, j], axis=j), name='Aexp_sum')
    out = tvm.compute((m, n), lambda i, j: Aexp[i, j] / Aexp_sum[i], name='out') 
    s = tvm.create_schedule(out.op)
    s[out].parallel(out.op.axis[0])
    print(tvm.lower(s, [A], simple_mode=True))
    mod = tvm.build(s, [A, out])
    return mod

dtype = 'float32'
N = 8
C = 10
x = np.random.normal(size=(N, C)).astype(dtype)

mod = get_tvm_softmax2d()
tvm_x = tvm.nd.array(x)
tvm_out = tvm.nd.empty(tvm_x.shape)
mod(tvm_x, tvm_out)

mx_x = mx.nd.array(x)
mx_out = mx.nd.softmax(mx_x, axis=-1)
np.testing.assert_almost_equal(tvm_out.asnumpy(), mx_out.asnumpy())
print(tvm_out, mx_out)
