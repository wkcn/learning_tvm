import tvm
import numpy as np

def matmul(n, m, l):
    """Return the computing expression of matrix multiplication
    A : n x l matrix
    B : l x m matrix
    C : n x m matrix with C = A B
    """
    k = tvm.reduce_axis((0, l), name='k')
    A = tvm.placeholder((n, l), name='A')
    B = tvm.placeholder((l, m), name='B')
    C = tvm.compute((n, m),
            lambda x, y: tvm.sum(A[x, k] * B[k, y], axis=k),
            name='C')
    return A, B, C

n = 10
A, B, C = matmul(n, n, n)
s = tvm.create_schedule(C.op)
print(tvm.lower(s, [A, B], simple_mode=True))
mod = tvm.build(s, [A, B, C])

dtype = 'float32'
a = tvm.nd.array(np.random.uniform(size=(n, n)).astype(dtype))
b = tvm.nd.array(np.random.uniform(size=(n, n)).astype(dtype))
c = tvm.nd.empty(a.shape)
print(a.shape, b.shape, c.shape)
mod(a, b, c)

np.testing.assert_allclose(np.dot(a.asnumpy(), b.asnumpy()),
                           c.asnumpy(), atol=1e-5)
