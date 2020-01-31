'''
Like vector add, broadcast add is a `memory-bound` operator.

A good schedule needs to consider `multiple` performance-related factors together.
'''
import numpy as np
import tvm
import d2ltvm

target = 'llvm'
def default(n):
    A, B, C = d2ltvm.broadcast_add((n, 1), (n, n))
    s = tvm.create_schedule(C.op)
    return s, (A, B, C)

def good_schedule(n):
    s, (A, B, C) = default(n)
    x, y = C.op.axis
    s[C].parallel(x)
    s[C].vectorize(y)
    return s, (A, B, C)

s, args = good_schedule(64)
print(tvm.lower(s, args, simple_mode=True))

def bad_schedule(n):
    s, (A, B, C) = default(n)
    x, y = C.op.axis
    s[C].reorder(y, x)
    s[C].parallel(y)
    s[C].vectorize(x)
    return s, (A, B, C)

s, args = bad_schedule(64)
print(tvm.lower(s, args, simple_mode=True))
