import ast
import os
import numpy as np
import mxnet as mx
from PIL import Image
import tvm
from tvm import relay
import json

name = 'resnet18_v1'
graph_fn, mod_fn, params_fn = ['./'+name+ext for ext in ('.json','.tar','.params')]
target = 'llvm'

with open('../data/imagenet1k_labels.txt') as fin:
    labels = ast.literal_eval(fin.read())

image = Image.open('../data/cat.jpg').resize((224, 224))

def build_pretrained_model(name):
    model = getattr(mx.gluon.model_zoo.vision, name)(pretrained=True)
    print(len(model.features), model.output)

    # compile pre-trained model
    shape = (1, 3, 224, 224)
    relay_mod, relay_params = relay.frontend.from_mxnet(model, {'data': shape})

    with relay.build_config(opt_level=3):
        graph, mod, params = relay.build(relay_mod, target, params=relay_params)
    return graph, mod, params



def image_preprocessing(image):
    image = np.array(image) - np.array([123., 117., 104.])
    image /= np.array([58.395, 57.12, 57.375])
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]
    # (N, C, H, W)
    return image.astype('float32')

x = image_preprocessing(image)

# build pretrained model 
if os.path.exists(mod_fn):
    # If the compiled library exists, load it
    graph = open(graph_fn).read()
    mod = tvm.module.load(mod_fn)
    params = relay.load_param_dict(open(params_fn, 'rb').read())

else:
    graph, mod, params = build_pretrained_model(name) 

    # Save the compiled library
    mod.export_library(mod_fn)
    with open(graph_fn, 'w') as f:
        f.write(graph)
    with open(params_fn, 'wb') as f:
        f.write(relay.save_param_dict(params))

'''
The compiled module has three parts:
graph is a json string described the neural network,
mod is a library that contains all compiled operators used to run the inference,
params is a dictionary mapping parameter name to weights.
'''

# inference
ctx = tvm.context(target) 

rt = tvm.contrib.graph_runtime.create(graph, mod, ctx)
# Set parameters
rt.set_input(**params)
rt.run(data=tvm.nd.array(x))
# asychronization ?
scores = rt.get_output(0).asnumpy()[0]

a = scores.argsort()[-1:-6:-1]
for i in a:
    print(labels[i])
