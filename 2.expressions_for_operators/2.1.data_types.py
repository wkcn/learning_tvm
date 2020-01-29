import tvm
import numpy as np

n = 3
dtype = 'int32'
def vector_add_dtype(dtype):
    A = tvm.placeholder((n, ), dtype=dtype)
    B = tvm.placeholder((n, ), dtype=dtype)
    C = tvm.compute(A.shape, lambda i: A[i] + B[i].astype(dtype))
    print(A.dtype, B.dtype, C.dtype)
    s = tvm.create_schedule(C.op)
    return tvm.build(s, [A, B, C])

mod = vector_add_dtype(dtype)

a = tvm.nd.array(np.array([1, 2, 3], dtype=dtype))
b = tvm.nd.array(np.array([2, 3, 9], dtype=dtype))
c = tvm.nd.empty(a.shape, dtype=dtype)
mod(a, b, c)

print(c, c.dtype)
