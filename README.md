# Zeno

### This is the python implementation of the paper "Zeno: Distributed Stochastic Gradient Descent with Suspicion-based Fault-tolerance"

### Requirements

The following python packages needs to be installed by pip:

1. MXNET (we use GPU, thus mxnet-cu80 is preferred)
2. Gluon-CV
3. Numpy

The users can simply run the following commond in their own virtualenv:

```bash
pip install --no-cache-dir numpy mxnet-cu80 gluoncv
```

### Run the demo

#### Options:

| Option     | Desctiption | 
| ---------- | ----------- | 
|--batch_size 100| batch size of the workers|
|--lr 0.1| learning rate|
|--nworkers 20| number of workers|
|--nepochs 200| total number of epochs|
|--gpu | index of GPU to be used|
|--nbyz | number of faulty workers|
|--byz_type | type of failures, bitflip or labelflip|
|--aggregation | aggregation method, mean, median, krum, or zeno|
|--zeno_size 4 | batch size of Zeno, $n_r$ in the paper|
|--rho_ratio | in the paper, $\rho = \gamma / rho\_ratio$|
|--b | number of trimmed values, $b$ in the paper|
|--iid 1| -iid 1 means the wokers are training on IID data|
|--interval 10| log interval|
|--seed 337 | random seed|

* Train with 20 workers, 8 of them are faulty with bit-flipping failures, Zeno as aggregation:
```bash
python mxnet_cnn_cifar10_impl.py --gpu 0 --nepochs 200 --lr 0.05 --batch_size 100 --nworkers 20 --nbyz 8 --byz_type bitflip --rho 200 --b 12 --zeno_size 4 --aggregation zeno
```


More detailed commands/instructions can be found in the demo script *test_zeno_1.sh*

