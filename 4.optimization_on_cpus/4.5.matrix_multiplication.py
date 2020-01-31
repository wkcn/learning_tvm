import d2ltvm
import numpy as np
import tvm
import timeit
import time
from matplotlib import pyplot as plt

target = 'llvm'
def np_matmul_timer(n):
    timer = timeit.Timer(setup='import numpy as np\n'
                         'import d2ltvm\n'
                         'a, b, c = d2ltvm.get_abc(%s)' % str((n,n)),
                         stmt = 'np.dot(a, b, out=c)')
    return timer.timeit

sizes = 2**np.arange(5, 10, 1)
# sizes = 2**np.arange(1, 3, 1)
exe_times = [d2ltvm.bench_workload(np_matmul_timer(n)) for n in sizes]
np_gflops = 2 * sizes **3 / 1e9 / np.array(exe_times)

def bench_matmul_tvm(func, sizes, target):
    def workload(nrepeats):
        timer = mod.time_evaluator(mod.entry_name, ctx=ctx, number=nrepeats)
        return timer(a, b, c).mean * nrepeats
    times = []
    for n in sizes:
        s, (A, B, C) = func(int(n))
        mod = tvm.build(s, [A, B, C], target)
        ctx = tvm.context(target, 0)
        a, b, c = d2ltvm.get_abc((n, n), lambda x: tvm.nd.array(x, ctx=ctx))
        times.append(d2ltvm.bench_workload(workload))
    return 2 * sizes**3 / 1e9 / np.array(times)

def matmul(n, m, l):
    """Return the computing expression of matrix multiplication
    A : n x l matrix
    B : l x m matrix
    C : n x m matrix with C = A B
    """
    k = tvm.reduce_axis((0, l), name='k')
    A = tvm.placeholder((n, l), name='A')
    B = tvm.placeholder((l, m), name='B')
    C = tvm.compute((n, m),
            lambda x, y: tvm.sum(A[x, k] * B[k, y], axis=k),
            name='C')
    return A, B, C

def default(n):
    A, B, C = matmul(n, n, n)
    return tvm.create_schedule(C.op), (A, B, C)

s, args = default(64)
print(tvm.lower(s, args, simple_mode=True))

default_gflops = bench_matmul_tvm(default, sizes, target)

# Simply switching these two for-loops will make all elements read and written sequentially.

def reorder(n):
    s, (A, B, C) = default(n)
    (x, y), (k, ) = C.op.axis, C.op.reduce_axis
    # change the axes order from (x,y,k) to (x,k,y)
    s[C].reorder(x, k, y)
    return s, (A, B, C)

s, args = reorder(64)
print(tvm.lower(s, args, simple_mode=True))
reorder_gflops = bench_matmul_tvm(reorder, sizes, target)

def parallel(n):
    s, (A, B, C) = reorder(n)
    s[C].parallel(C.op.axis[0]) # x
    return s, (A, B, C)

s, args = parallel(64)
print(tvm.lower(s, args, simple_mode=True))
parallel_gflops = bench_matmul_tvm(parallel, sizes, target)

def vectorize(n):
    s, (A, B, C) = parallel(n)
    s[C].vectorize(C.op.axis[1]) # y
    return s, (A, B, C)

s, args = vectorize(64)
print(tvm.lower(s, args, simple_mode=True))
vectorize_gflops = bench_matmul_tvm(vectorize, sizes, target)

d2ltvm.plot_gflops(sizes, [np_gflops, default_gflops, reorder_gflops, parallel_gflops, vectorize_gflops],
            ['numpy', 'default', 'reorder', 'parallel', 'vectorize'])
plt.show()
