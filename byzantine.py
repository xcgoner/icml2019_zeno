import mxnet as mx
from mxnet import nd

# no faulty workers
def no_byz(v, f):
    pass

# failures that add Gaussian noise
def gaussian_attack(v, f):
    for i in range(f):
        v[i] = mx.nd.random.normal(0, 200, shape=v[i].shape)

# bit-flipping failure
def bitflip_attack(v, f):
    for i in range(f):
        if i > 0:
            v[i][:] = -v[0]
    v[0][:] = -v[0]

# label-flipping failure is implemented in mxnet_cnn_cifar10_impl.py