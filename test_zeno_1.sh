#!/bin/bash

export MXNET_CUDNN_AUTOTUNE_DEFAULT=0

export CUDA_VISIBLE_DEVICES=1

stdbuf -o 0  python mxnet_cnn_cifar10_impl.py --gpu 0 --nepochs 200 --lr 0.05 --batch_size 100 --nworkers 20 --nbyz 8 --byz_type bitflip --rho 200 --b 12 --zeno_size 4 --aggregation zeno








