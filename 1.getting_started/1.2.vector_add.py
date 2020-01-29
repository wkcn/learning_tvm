import tvm
import numpy as np

def vector_add(n):
    A = tvm.placeholder((n, ), name='a')
    B = tvm.placeholder((n, ), name='b')
    C = tvm.compute(A.shape, lambda i: A[i] + B[i], name='c')
    return A, B, C

n = 3
A, B, C = vector_add(n)

# create a schedule
s = tvm.create_schedule(C.op)

# A schedule consists of several stages
# Each stage corresponds to an operation to describe how it is scheduled. We can access a particular stage by either s[C] or s[C.op].

'''
The lower method accepts the schedule and input and output tensors.
The simple_mode=True will print the program in a simple and compact way.
'''
print(tvm.lower(s, [A, B, C], simple_mode=True))

mod = tvm.build(s, [A, B, C])

# The default data type in TVM is float32
dtype = np.float32
a = tvm.nd.array(np.array([1, 2, 3], dtype=dtype))
b = tvm.nd.array(np.array([4, 5, 6], dtype=dtype))
c = tvm.nd.empty(b.shape)
mod(a, b, c)
print(c)
