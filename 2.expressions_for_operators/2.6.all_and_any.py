'''
tvm.all
tvm.any
'''
import numpy as np
import tvm

print(any((0, 1, 2)), all((0, 1, 2)))

a = np.ones((3, 4), dtype='float32')
# applying a zero padding of size 1 to a
b = np.zeros((5, 6), dtype='float32')
b[1:-1,1:-1] = a
print(b)

p = 1
n, m = tvm.var('n'), tvm.var('m')
A = tvm.placeholder((n, m), name='a')
B = tvm.compute((n + p * 2, m + p * 2), lambda i, j: tvm.if_then_else(
        tvm.any(i < p, i >= n + p, j < p, j >= m + p),
        0,
        A[i-p, j-p]
    ), name='b')

s = tvm.create_schedule(B.op)
print(tvm.lower(s, [A, B], simple_mode=True))

mod = tvm.build(s, [A, B])

c = tvm.nd.array(np.empty_like(b))
mod(tvm.nd.array(a), c)
print(c)
