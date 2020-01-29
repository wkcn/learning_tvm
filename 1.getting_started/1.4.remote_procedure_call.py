'''
Setup the remote machine
python -m tvm.exec.rpc_server --host 0.0.0.0 --port=9090
'''

import numpy as np
import tvm
from tvm import rpc

target_url = '0.0.0.0'
target_port = 9090
target = 'llvm -target=x86_64-pc-linux-gnu'
n = 3
dtype = np.float32

def vector_add(n):
    A = tvm.placeholder((n, ), name='a')
    B = tvm.placeholder((n, ), name='b')
    C = tvm.compute(A.shape, lambda i: A[i] + B[i], name='c')
    return A, B, C

A, B, C = vector_add(n)

s = tvm.create_schedule(C.op)

mod = tvm.build(s, [A, B, C])

mod_fname = 'vector-add.tar'
mod.export_library(mod_fname)

if target_url != '0.0.0.0':
    remote = rpc.connect(url=target_url, port=target_port)
else:
    remote = rpc.LocalSession()
# Even if running a pretrained model in remote machine, it only need upload `mod_fname`
remote.upload(mod_fname)
remote_mod = remote.load_module(mod_fname)

ctx = remote.cpu()
a = tvm.nd.array(np.array([1, 2, 3], dtype=dtype), ctx=ctx)
b = tvm.nd.array(np.array([4, 5, 6], dtype=dtype), ctx=ctx)
c = tvm.nd.empty(b.shape)

remote_mod(a, b, c)
print(c.asnumpy())
