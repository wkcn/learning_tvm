'''
The default scheduling generates naive `single-thread` CPU program.
`Parallelization` improves performance for large workloads.
We can split a for-loop and then `vectorize` the inner loop if the system supports SIMD.
'''
import tvm
import numpy as np

def vector_add_default():
    n = tvm.var('n')
    A = tvm.placeholder((n, ), name='A')
    B = tvm.placeholder((n, ), name='B')
    C = tvm.compute((n, ), lambda i: A[i] + B[i], name='C')
    return A, B, C

target = 'llvm'
A, B, C = vector_add_default()
s = tvm.create_schedule(C.op)

# <class 'tvm.container.Array'> <class 'tvm.schedule.IterVar'>
print(type(C.op.axis), type(C.op.axis[0]))

# Parallelization
# pass tvm.schedule.IterVar
s[C].parallel(C.op.axis[0])

print(tvm.lower(s, [A, B], simple_mode=True))

mod = tvm.build(s, [A, B, C], target) 
print(mod.get_source()[:500])
'''
parallelization significantly improves the performance when the workloads are large, e.g. vector lengths beyond 10^4.
However, the parallelization overhead impact the performance for small workloads, where single thread is even faster.
The performance drops at a larger size as multi-core comes in play, leading to a larger amount of L2 cache in total.
'''

# Vectorization
def vectorized_vector_add():
    # use SIMD
    A, B, C = vector_add_default()
    s = tvm.create_schedule(C.op)
    outer, inner = s[C].split(C.op.axis[0], factor=8)
    # <class 'tvm.schedule.IterVar'> <class 'tvm.schedule.IterVar'>
    print(type(outer), type(inner))
    s[C].parallel(outer)
    s[C].vectorize(inner)
    return s, (A, B, C)
s, args = vectorized_vector_add()
print(tvm.lower(s, args, simple_mode=True))
