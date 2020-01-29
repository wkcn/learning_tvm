import tvm
import numpy as np

n = tvm.var('n')
m = tvm.var('m')

A = tvm.placeholder((n, m), name='a')

# Matrix Transpose
def matrix_transpose():
    B = tvm.compute((m, n), lambda i, j: A[j, i], 'b')
    s = tvm.create_schedule(B.op)

    print(tvm.lower(s, [A, B], simple_mode=True))

    a = np.arange(12, dtype='float32').reshape((3, 4))
    b = np.empty((4, 3), dtype='float32')
    a, b = tvm.nd.array(a), tvm.nd.array(b)

    mod = tvm.build(s, [A, B])
    mod(a, b)
    print(a, b)

# Reshaping

# flatten
# (m, n) -> k
def flatten():
    C = tvm.compute((m*n, ), lambda k: A[k // m, k % m]) 
    s = tvm.create_schedule(C.op)
    print(tvm.lower(s, [A, C], simple_mode=True))

# General reshape
def reshape():
    p, q = tvm.var('p'), tvm.var('q')
    # (m, n) -> (p, q)
    D = tvm.compute((p, q), lambda i, j: A[(i * q + j) // m, (i * q + j) % m], name='d')

    s = tvm.create_schedule(D.op)
    print(tvm.lower(s, [A, D], simple_mode=True))

    def func(i, j):
        k = i * q + j
        return A[k // m, k % m]
    E = tvm.compute((p, q), func, name='e')
    s = tvm.create_schedule(E.op)
    print(tvm.lower(s, [A, E], simple_mode=True))

    mod = tvm.build(s, [A, E])
    a = np.arange(12, dtype='float32').reshape((3,4))
    b = np.zeros((5, 4), dtype='float32')
    a, b = tvm.nd.array(a), tvm.nd.array(b)
    print(a.shape, b.shape)

    mod(a, b)
    print(a, a.dtype, b, b.dtype)
    print(a.asnumpy().astype('int'), b.asnumpy().astype('int')[:3, :4])

# Slice
def test_slice():
    # a[bi::si, bj::sj], si and sj are both strides
    # pass the variables bi, bj, si and sj as arguments
    '''
    Both shape dimensions and indices can be expressions with variables.
    If a variable doesn’t only appear in the shape tuple, we need to pass it as an argument when compiling.
    '''
    names = ['bi', 'si', 'bj', 'sj']
    bi, si, bj, sj = map(tvm.var, names)
    # A(n, m)
    shape = ((n-bi)//si, (m-bj)//sj)
    B = tvm.compute(shape, lambda i, j: A[bi+si*i, bj+sj*j], name='b')
    s = tvm.create_schedule((B.op))
    print(tvm.lower(s, [A, B, bi, si, bj, sj], simple_mode=True))
    mod = tvm.build(s, [A, B, bi, si, bj, sj])

    a = tvm.nd.array(np.random.normal(size=(3, 4)).astype('float32'))
    b = tvm.nd.array(np.empty((1, 3), dtype='float32'))

    print(a.shape, b.shape)
    mod(a, b, 1, 2, 1, 1)
    np.testing.assert_equal(b.asnumpy(), a.asnumpy()[1::2, 1::1])

    b = tvm.nd.array(np.empty((1, 2), dtype='float32'))
    mod(a, b, 2, 1, 0, 2)
    np.testing.assert_equal(b.asnumpy(), a.asnumpy()[2::1, 0::2])

if __name__ == '__main__':
    matrix_transpose()
    flatten()
    reshape()
    test_slice()
