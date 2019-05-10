import mxnet as mx
from mxnet import nd, autograd, gluon
import numpy as np
import math

def no_byz(v, f):
    pass

def marginal_median(gradients, net, lr, f = 0, byz = no_byz):
    # X is a 2d list of nd array
    # TODO: improve the implementation of median, the current one is very slow
    param_list = [nd.concat(*[xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]
    byz(param_list, f)
    sorted_array = nd.sort(nd.concat(*param_list, dim=1), axis=-1)
    if sorted_array.shape[-1] % 2 == 1:
        median_nd = sorted_array[:, sorted_array.shape[-1]/2]
    else:
        median_nd = (sorted_array[:, (sorted_array.shape[-1]/2-1)] + sorted_array[:, (sorted_array.shape[-1]/2)]) / 2.
    # np_array = nd.concat(*param_list, dim=1).asnumpy()
    # median_nd = nd.array(np.median(np_array, axis=1))
    idx = 0
    for j, (param) in enumerate(net.collect_params().values()):
        if param.grad_req != 'null':
            # mapping back to the collection of ndarray
            param.set_data(param.data() - lr * median_nd[idx:(idx+param.data().size)].reshape(param.data().shape))
            idx += param.data().size
    
def simple_mean(gradients, net, lr, f = 0, byz = no_byz):
    # X is a 2d list of nd array
    param_list = [nd.concat(*[xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]
    byz(param_list, f)
    mean_nd = nd.mean(nd.concat(*param_list, dim=1), axis=-1)
    idx = 0
    for j, (param) in enumerate(net.collect_params().values()):
        if param.grad_req != 'null':
            # mapping back to the collection of ndarray
            param.set_data(param.data() - lr * mean_nd[idx:(idx+param.data().size)].reshape(param.data().shape))
            idx += param.data().size

# score function used by Krum
def score(gradient, v, f):
    if 2*f+2 > v.shape[1]:
        f = int(math.floor((v.shape[1]-2)/2.0))
    num_neighbours = v.shape[1] - 2 - f
    sorted_distance = nd.square(v - gradient).sum(axis=0).sort()
    return nd.sum(sorted_distance[1:(1+num_neighbours)]).asscalar()

def krum(gradients, net, lr, f = 0, byz = no_byz):
    # X is a 2d list of nd array
    param_list = [nd.concat(*[xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]
    byz(param_list, f)
    v = nd.concat(*param_list, dim=1)
    scores = nd.array([score(gradient, v, f) for gradient in param_list])
    min_idx = int(scores.argmin(axis=0).asscalar())
    krum_nd = nd.reshape(param_list[min_idx], shape=(-1,))
    idx = 0
    for j, (param) in enumerate(net.collect_params().values()):
        if param.grad_req != 'null':
            # mapping back to the collection of ndarray
            param.set_data(param.data() - lr * krum_nd[idx:(idx+param.data().size)].reshape(param.data().shape))
            idx += param.data().size

def zeno(gradients, net, loss_fun, lr, sample, rho_ratio, b, f = 0, byz = no_byz):
    # X is a 2d list of nd array
    param_list = [nd.concat(*[xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]

    param_net = [xx.data().copy() for xx in net.collect_params().values()]

    byz(param_list, f)
    output = net(sample[0])
    loss_1_nd = loss_fun(output, sample[1])
    loss_1 = nd.mean(loss_1_nd).asscalar()
    scores = []
    rho = lr / rho_ratio
    for i in range(len(param_list)):
        idx = 0
        for j, param in enumerate(net.collect_params().values()):
            if param.grad_req != 'null':
                param.set_data(param_net[j] - lr * param_list[i][idx:(idx+param.data().size)].reshape(param.data().shape))
                idx += param.data().size
        output = net(sample[0])
        loss_2_nd = loss_fun(output, sample[1])
        loss_2 = nd.mean(loss_2_nd).asscalar()
        scores.append(loss_1 - loss_2 - rho * param_list[i].square().mean().asscalar())
    scores_np = np.array(scores)
    scores_idx = np.argsort(scores_np)
    scores_idx = scores_idx[-(len(param_list)-b):].tolist()
    g_aggregated = nd.zeros_like(param_list[0])
    for idx in scores_idx:
        g_aggregated += param_list[idx]
    g_aggregated /= float(len(scores_idx))
    
    idx = 0
    for j, (param) in enumerate(net.collect_params().values()):
        if param.grad_req != 'null':
            # mapping back to the collection of ndarray
            param.set_data(param_net[j] - lr * g_aggregated[idx:(idx+param.data().size)].reshape(param.data().shape))
            idx += param.data().size