from __future__ import print_function
import nd_aggregation
import mxnet as mx
from mxnet import nd, autograd, gluon
from mxnet.gluon.data.vision import transforms
from gluoncv.data import transforms as gcv_transforms
import numpy as np
import time, random, argparse, itertools
import byzantine

np.warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
# parser.add_argument("--net", help="net", type=str)
parser.add_argument("--batch_size", help="batch size", type=int, default=100)
parser.add_argument("--lr", help="float", type=float)
parser.add_argument("--nworkers", help="# workers", type=int)
parser.add_argument("--nepochs", help="# epochs", type=int)
parser.add_argument("--gpu", help="index of gpu", type=int)
parser.add_argument("--nbyz", help="# byzantines", type=int)
parser.add_argument("--byz_type", help="type of failure", type=str)
parser.add_argument("--aggregation", help="aggregation", type=str)
parser.add_argument("--zeno_size", help="zeno batch size", type=int, default=4)
# \rho in the paper is equivalent to lr / rho_ratio in the code
parser.add_argument("--rho_ratio", help="ratio to learning rate", type=float)
parser.add_argument("--b", help="b, number of trimmed values", type=int)
parser.add_argument("--iid", help="iid data or not", type=int, default=1)
parser.add_argument("--interval", help="log interval", type=int, default=5)
parser.add_argument("--seed", help="random seed", type=int, default=0)
args = parser.parse_args()

import sys
print(' '.join(sys.argv))

if args.gpu == -1:
    ctx = mx.cpu()
else:
    ctx = mx.gpu(args.gpu)


with mx.gpu(args.gpu):

    batch_size = args.batch_size

    classes = 10

    # cnn, lr=.1
    net = gluon.nn.Sequential()
    with net.name_scope():
        #  First convolutional layer
        net.add(gluon.nn.Conv2D(channels=64, kernel_size=3, padding=(1,1), activation='relu'))
        net.add(gluon.nn.BatchNorm())
        net.add(gluon.nn.Conv2D(channels=64, kernel_size=3, padding=(1,1), activation='relu'))
        net.add(gluon.nn.BatchNorm())
        net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        net.add(gluon.nn.Dropout(rate=0.25))
        #  Second convolutional layer
        # net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        # Third convolutional layer
        net.add(gluon.nn.Conv2D(channels=128, kernel_size=3, padding=(1,1), activation='relu'))
        net.add(gluon.nn.BatchNorm())
        net.add(gluon.nn.Conv2D(channels=128, kernel_size=3, padding=(1,1), activation='relu'))
        net.add(gluon.nn.BatchNorm())
        net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        net.add(gluon.nn.Dropout(rate=0.25))
        # net.add(gluon.nn.Conv2D(channels=64, kernel_size=3, padding=(1,1), activation='relu'))
        # net.add(gluon.nn.Conv2D(channels=64, kernel_size=3, padding=(1,1), activation='relu'))
        # net.add(gluon.nn.Conv2D(channels=64, kernel_size=3, padding=(1,1), activation='relu'))
        # net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        # Flatten and apply fullly connected layers
        net.add(gluon.nn.Flatten())
        # net.add(gluon.nn.Dense(512, activation="relu"))
        # net.add(gluon.nn.Dense(512, activation="relu"))
        net.add(gluon.nn.Dense(128, activation="relu"))
        # net.add(gluon.nn.Dense(256, activation="relu"))
        net.add(gluon.nn.Dropout(rate=0.25))
        net.add(gluon.nn.Dense(classes))

    # byzantine
    if args.byz_type == 'label':
        byz = byzantine.no_byz
    else:
        if args.byz_type == 'bitflip':
            byz = byzantine.bitflip_attack
        else:
            byz = byzantine.no_byz

    zeno_batch_size = args.zeno_size

    if args.iid == 1:
        shuffle_data = True
    else:
        shuffle_data = False

    def transform(data, label):
        data = mx.nd.transpose(data, (2,0,1))
        data = data.astype(np.float32) / 255
        return data, label

    acc_top1 = mx.metric.Accuracy()
    acc_top5 = mx.metric.TopKAccuracy(5)
    train_cross_entropy = mx.metric.CrossEntropy()

    # set random seed
    mx.random.seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # data loader
    train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.CIFAR10('/data/cx2', train=True, transform=transform),
                                batch_size, shuffle=shuffle_data, last_batch='discard')
    val_train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.CIFAR10('/data/cx2', train=True, transform=transform),
                                batch_size, shuffle=False, last_batch='keep')
    val_test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.CIFAR10('/data/cx2', train=False, transform=transform),
                                batch_size, shuffle=False, last_batch='keep')
    zeno_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.CIFAR10('/data/cx2', train=True, transform=transform),
                                    zeno_batch_size, shuffle=True, last_batch='rollover')

    zeno_iter = itertools.cycle(zeno_data)

    # initialization 
    net.initialize(mx.init.Xavier(), ctx=ctx, force_reinit=True)

    # loss function
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

    num_workers = args.nworkers
    lr = args.lr / batch_size

    epochs = args.nepochs
    itr = 0
    grad_list = []
    worker_idx = 0
    time_0 = time.time()
    for e in range(epochs):
        
        tic = time.time()
        # training
        for i, (data, label) in enumerate(train_data):
            # label-flipping failures
            if args.byz_type == 'label' and worker_idx < args.nbyz:
                label = 9 - label
            with autograd.record():
                output = net(data)
                loss = softmax_cross_entropy(output, label)
            loss.backward() 

            grad_collect = []
            for param in net.collect_params().values():
                if param.grad_req != 'null':
                    grad_collect.append(param.grad().copy())
            grad_list.append(grad_collect)

            # nd.waitall()

            itr += 1
            worker_idx += 1

            if itr % num_workers == 0:
                # aggregate
                nd.waitall()
                worker_idx = 0
                if args.aggregation == 'median':
                    nd_aggregation.marginal_median(grad_list, net, lr, args.nbyz, byz)
                elif args.aggregation == 'krum':
                    nd_aggregation.krum(grad_list, net, lr, args.nbyz, byz)
                elif args.aggregation == 'mean':
                    nd_aggregation.simple_mean(grad_list, net, lr, args.nbyz, byz)
                elif args.aggregation == 'zeno':
                    zeno_sample = zeno_iter.next()
                    nd_aggregation.zeno(grad_list, net, softmax_cross_entropy, lr, zeno_sample, args.rho_ratio, args.b, args.nbyz, byz)
                else:
                    nd_aggregation.simple_mean(grad_list, net, lr, args.nbyz, byz)

                del grad_list
                grad_list = []
        nd.waitall()
        toc = time.time()

        if e % args.interval == 0 :
            acc_top1.reset()
            acc_top5.reset()
            train_cross_entropy.reset()
            # accuracy on testing data
            for i, (data, label) in enumerate(val_test_data):
                outputs = net(data)
                acc_top1.update(label, outputs)
                acc_top5.update(label, outputs)
            # cross entropy on training data
            for i, (data, label) in enumerate(val_train_data):
                outputs = net(data)
                train_cross_entropy.update(label, nd.softmax(outputs))

            _, top1 = acc_top1.get()
            _, top5 = acc_top5.get()
            _, crossentropy = train_cross_entropy.get()

            print('[Epoch %d] validation: acc-top1=%f acc-top5=%f, loss=%f, epoch_time=%f, elapsed=%f' % (e, top1, top5, crossentropy, toc-tic, time.time()-time_0))
            