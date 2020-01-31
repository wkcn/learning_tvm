'''
Blocked tiling improves cache efficiency for matrix multiplication.

Data to be frequently read and written should be placed in a buffer explicitly to reduce cache misses.
'''
import tvm
import d2ltvm
import numpy as np

tx, ty, tk = 32, 32, 4

def block(n):
    A, B, C = d2ltvm.matmul(n, n, n)
    s = tvm.create_schedule(C.op)

    # Tile by blocks, and then parallelize the computation of each block
    '''
    tile(x_parent, y_parent, x_factor, y_factor) method of tvm.schedule.Stage instance
    Perform tiling on two dimensions

    The final loop order from outmost to inner most are
    [x_outer, y_outer, x_inner, y_inner]
    '''
    xo, yo, xi, yi = s[C].tile(*C.op.axis, tx, ty)

    '''
    fuse(*args) method of tvm.schedule.Stage instance
    Fuse multiple consecutive iteration variables into a single iteration variable.

    fused = fuse(...fuse(fuse(args[0], args[1]), args[2]),..., args[-1])
    The order is from outer to inner.
    '''
    xy = s[C].fuse(xo, yo)

    s[C].parallel(xy)

    # Optimize the computation of each block
    ko, ki = s[C].split(s[C].op.reduce_axis[0], factor=tk) 
    s[C].reorder(ko, xi, ki, yi)
    s[C].vectorize(yi)
    # unroll the for-loop of ki
    s[C].unroll(ki)
    return s, (A, B, C)

s, (A, B, C) = block(64)
print(tvm.lower(s, [A, B, C], simple_mode=True))

# The non-continuous write issue is severer than the non-continuous read.
# we need to use s[Cached] instead of s[C] to optimize the submatrix computation.
def cached_block(n):
    A, B, C = d2ltvm.matmul(n, n, n)
    s = tvm.create_schedule(C.op)
    # Create a write cache for C
    CachedC = s.cache_write(C, 'local')
    # Same as before, first tile by blocks, and then parallelize the
    # computation of each block
    xo, yo, xi, yi = s[C].tile(*C.op.axis, tx, ty)
    xy = s[C].fuse(xo, yo)
    s[C].parallel(xy)
    # Use the write cache for the output of the xy axis, namely a block.
    s[CachedC].compute_at(s[C], xy)
    # Same as before to optimize the computation of a block .
    xc, yc = s[CachedC].op.axis
    ko, ki = s[CachedC].split(CachedC.op.reduce_axis[0], factor=tk)
    s[CachedC].reorder(ko, xc, ki, yc)
    s[CachedC].unroll(ki)
    s[CachedC].vectorize(yc)
    return s, (A, B, C)

s, (A, B, C) = cached_block(512)
print(tvm.lower(s, [A, B, C], simple_mode=True))
