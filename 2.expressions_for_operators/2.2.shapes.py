import tvm
import numpy as np

n = tvm.var(name='n')
print(type(n), n.dtype)

A = tvm.placeholder((n, ), name='a')
B = tvm.placeholder((n, ), name='b')
C = tvm.compute(A.shape, lambda i: A[i] + B[i], name='c')
print(A.name, B.name, C.name)

s = tvm.create_schedule(C.op)
print(s, s[C], s[A])
print(tvm.lower(s, [A, B, C], simple_mode=True))

mod = tvm.build(s, [A, B, C])

dtype = 'float32'
a = tvm.nd.array(np.array([1, 2, 3], dtype=dtype))
b = tvm.nd.array(np.array([4, 7, 3], dtype=dtype))
c = tvm.nd.empty(a.shape)

mod(a, b, c)
print(c)

# Multi-dimensional shape 
def vector_add_nd(ndim):
    A = tvm.placeholder([tvm.var() for _ in range(ndim)])
    B = tvm.placeholder(A.shape)
    C = tvm.compute(A.shape, lambda *i: A[i] + B[i]) # *i
    s = tvm.create_schedule(C.op)
    print(tvm.lower(s, [A, B, C], simple_mode=True))
    return tvm.build(s, [A, B, C])

mod = vector_add_nd(2)
a = tvm.nd.array(np.array([[1, 2, 3], [4, 5, 6]], dtype=dtype))
b = tvm.nd.array(np.array([[4, 7, 3], [2, 3, 4]], dtype=dtype))
c = tvm.nd.empty(a.shape)
mod(a, b, c)
print(c)
