import tvm
import numpy as np

# lower triangle
a = np.arange(12, dtype='float32').reshape((3, 4))
print(np.tril(a))

n, m = tvm.var('n'), tvm.var('m')
A = tvm.placeholder((n, m), name='a')
B = tvm.compute(A.shape, lambda i, j: tvm.if_then_else(i >= j, A[i, j], 0.0))

b = tvm.nd.array(np.empty_like(a))
s = tvm.create_schedule(B.op)
print(tvm.lower(s, [A, B], simple_mode=True))
mod = tvm.build(s, [A, B])
mod(tvm.nd.array(a), b)
print(b)
