import tvm
import numpy as np

def broadcast_add(shape1, shape2):
    assert len(shape1) == 2 and len(shape2) == 2
    for i in range(len(shape1)):
        assert shape1[i] == shape2[i] or shape1[i] == 1 or shape2[i] == 1
    A = tvm.placeholder(shape1, name='A')
    B = tvm.placeholder(shape2, name='B')
    m = shape1[0] if shape2[0] == 1 else shape2[0]
    n = shape1[1] if shape2[1] == 1 else shape2[1]

    def f(x, y):
        # the type of `x` is `tvm.expr.Var`
        # the type of shape is a list of int
        ai = 0 if shape1[0] == 1 else x
        aj = 0 if shape1[1] == 1 else y

        bi = 0 if shape2[0] == 1 else x
        bj = 0 if shape2[1] == 1 else y

        return A[ai, aj] + B[bi, bj]
    C = tvm.compute((m, n), f, name='C') 
    return A, B, C

m, n = 3, 4
shape1 = (m, 1)
shape2 = (m, n)
A, B, C = broadcast_add(shape1, shape2)
s = tvm.create_schedule(C.op)
print(tvm.lower(s, [A, B], simple_mode=True))

mod = tvm.build(s, [A, B, C])

def get_bcast_data(shape1, shape2, constructor=None):
    """Return random tensors a, b
    and empty tensor c to store broadcast results between a and b

    shape1, shape2: shapes of input tensors
    constructor : user-defined tensor constructor
    """
    np.random.seed(0)
    a = np.random.normal(size=shape1).astype("float32")
    b = np.random.normal(size=shape2).astype("float32")
    out_shape = (shape1[0] if shape2[0] == 1 else shape2[0],
                 shape1[1] if shape2[1] == 1 else shape2[1])
    c = np.empty(out_shape, dtype='float32')
    if constructor:
        a, b, c = [constructor(x) for x in (a, b, c)]
    return a, b, c

a, b, c = get_bcast_data(shape1, shape2, tvm.nd.array)
mod(a, b, c)

np.testing.assert_allclose(np.add(a.asnumpy(), b.asnumpy()), c.asnumpy(), atol=1e-5)
print(a.shape, b.shape, c.shape)
