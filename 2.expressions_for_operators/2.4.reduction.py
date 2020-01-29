'''
tvm.reduce_axis
tvm.comm_reducer
'''
import tvm
import numpy as np

n, m = tvm.var('n'), tvm.var('m')

# sum(axis=1)
'''
    for i in range(n):
        b[i] = np.sum(a[i,:])
'''
A = tvm.placeholder((n, m), name='a')
j = tvm.reduce_axis((0, m), name='j')
B = tvm.compute((n, ), lambda i: tvm.sum(A[i, j], axis=j), name='b')

s = tvm.create_schedule(B.op)

print(tvm.lower(s, [A, B], simple_mode=True))

a = np.random.normal(size=(3,4)).astype('float32')
b = a.sum(axis=1)
mod = tvm.build(s, [A, B])
c = tvm.nd.array(np.empty((3,), dtype='float32'))
mod(tvm.nd.array(a), c)
np.testing.assert_equal(b, c.asnumpy())

# (0, n) is the range of the domain of iteration
i = tvm.reduce_axis((0, n), name='i')
B = tvm.compute((), lambda: tvm.sum(A[i, j], axis=(i, j)), name='b')
s = tvm.create_schedule(B.op)
print(tvm.lower(s, [A, B], simple_mode=True))

mod = tvm.build(s, [A, B])
c = tvm.nd.array(np.empty((), dtype='float32'))
mod(tvm.nd.array(a), c)
np.testing.assert_allclose(a.sum(), c.asnumpy(), atol=1e-5)

# Commutative Reduction
# f(a, b) = f(b, a) 

# prod(axis=1)
comp = lambda a, b: a * b
init = lambda dtype: tvm.const(1, dtype=dtype)

product = tvm.comm_reducer(comp, init)

n, m = tvm.var('n'), tvm.var('m')
A = tvm.placeholder((n, m), name='a')
k = tvm.reduce_axis((0, m), name='k')
B = tvm.compute((n, ), lambda i: product(A[i, k], axis=k), name='b')
s = tvm.create_schedule(B.op)
print(tvm.lower(s, [A, B], simple_mode=True))

mod = tvm.build(s, [A, B])
b = tvm.nd.array(np.empty((3,), dtype='float32'))
mod(tvm.nd.array(a), b)
np.testing.assert_allclose(a.prod(axis=1), b.asnumpy(), atol=1e-5)
