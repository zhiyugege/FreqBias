#!/usr/bin/env python

import os
import argparse
import torch as ch
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from robustness import model_utils, datasets, train, defaults
from cox.utils import Parameters
import cox.store
from data_utils import *


parser = argparse.ArgumentParser(description='train configs')
parser.add_argument('--dataset' ,help='the used dataset')
parser.add_argument('--arch' , default='resnet18', help='the used model')
parser.add_argument('--freq', default=None, help='frequecy')
parser.add_argument('--r', default=0, help='radius of frequecy domain')
parser.add_argument('--lr', default=0.1, help='leanring rate')
parser.add_argument('--epoch', default=200, help='epoch')
parser.add_argument('--task', default=None, help='expriment id')
parser.add_argument('--gpu', default="0", help='choose gpu to train')
args = parser.parse_args()

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

DATA_PATH_DICT = {'cifar10': '/data/cifar','cifar100':'/data/cifar','resIN':'/data/ILSVRC2012'}
DATA_NAME_DICT = {'cifar10':'CIFAR', 'cifar100':'CIFAR100','resIN':'RestrictedImageNet'}
DEVICES_ID = None
if len(args.gpu)>1: 
    DEVICES_ID = [int(id) for id in args.gpu.split(',')]
# configs
ARCH = args.arch
FREQ  = args.freq
RADIUS = args.r
DATA = args.dataset
OUT_DIR = args.dataset
LR = float(args.lr)
EPOCH = int(args.epoch)
NUM = int(args.number)

dataset_function = getattr(datasets, DATA_NAME_DICT[DATA])
dataset = dataset_function(DATA_PATH_DICT[DATA])


# DATA loader
print("preparing data {}...".format(DATA))
train_loader, test_loader = get_dataloader(DATA, FREQ, RADIUS)
# train_loader, test_loader = dataset.make_loaders(workers=8, batch_size=128)
print('data loader is ok :)')


# prepare model
print("preparing model {}...".format(ARCH))
model_kwargs = {
    'arch': ARCH,
    'dataset': dataset
}
model, _ = model_utils.make_and_restore_model(**model_kwargs)
print('model is ok :)')

#train
train_kwargs = {
    'out_dir'  : OUT_DIR,
    'epochs'   : EPOCH,
    'lr'       : LR,
    'momentum' : 0.9,
    'weight_decay': 5e-4,
    'lr_interpolation ': 'step',
    'custom_lr_multplier ': '[(60,0.1),(120,0.1),(180,0.1)]',
    'save_ckpt_iters' : 1,
    'adv_train': 0,
    'constraint': '2',
    'eps': 0.5,
    'attack_lr': 0.1,
    'attack_steps':7,
    'use_best': True,
    'random_restarts': 0
}

train_args = Parameters(train_kwargs)
train_args = defaults.check_and_fill_args(train_args,
                        defaults.TRAINING_ARGS, dataset_function)

#log
if args.task==None:
    if FREQ==None:
        task = ARCH+'_baseline'
    else:
        task = ARCH+'_'+FREQ+'_'+RADIUS
else:
    task = args.task

out_store = cox.store.Store(OUT_DIR, exp_id = task)

print("="*20)
if FREQ:
    print("Ready to train  {}  on dataset  {} ,frequency : {} and r: {}".format(ARCH,DATA,FREQ,RADIUS))
else:
    print("Ready to train  {}  on dataset  {} , baseline".format(ARCH,DATA))
print("="*20)
train.train_model(train_args, model, (train_loader, test_loader), store=out_store, dp_device_ids=DEVICES_ID)  
