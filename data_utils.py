#!/usr/bin/env python
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision
import numpy as np
import random
import copy
from tiny_imagenet import *

BATCH_SIZE = 128
NUM_WORKERS = 8

def build_dataset(dataset, freq, r):

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                          (4, 4, 4, 4), mode='reflect').squeeze()),
        transforms.ToPILImage(),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    transform_im_train = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])
    if dataset == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10(root='/data/cifar', train=True, download=False, transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR10(root='/data/cifar', train=False, transform=transform_test)
        num_classes = 10

    if dataset == 'cifar100':
        train_dataset = torchvision.datasets.CIFAR100(root='/data/cifar', train=True, download=False, transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR100(root='/data/cifar', train=False, transform=transform_test)
        num_classes = 100
   
    if r:
        freq_file = '/data/'+dataset+'/hybrid_high_'+str(r)+'.npy'
        label_file = '/data/'+dataset+'/train_label.npy'
        print('loading frequency image file from '+freq_file)
        print('loading frequency label file from '+label_file)
        freq_data, freq_label = read_freq_file(freq_file,label_file)
        train_dataset.data = freq_data
        train_dataset.targets = freq_label
    if freq:
        freq_file = '/data/'+dataset+'/dft/test_data_'+freq+'_'+str(r)+'.npy'
        label_file = '/data/'+dataset+'/dft/test_label.npy'
        print('loading frequency image file from '+freq_file)
        print('loading frequency label file from '+label_file)
        freq_data, freq_label = read_freq_file(freq_file,label_file)
        test_dataset.data = freq_data
        test_dataset.targets = freq_label
        
    return train_dataset, test_dataset
    




def get_dataloader(dataset, freq, r):
    
    train_dataset, test_dataset = build_dataset(dataset, freq, r)

    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=BATCH_SIZE, 
                                               shuffle=True, 
                                               num_workers=NUM_WORKERS)
    test_loader = torch.utils.data.DataLoader(test_dataset, 
                                              batch_size=BATCH_SIZE, 
                                              shuffle=False, 
                                              num_workers=NUM_WORKERS)

    return train_loader,test_loader




def read_freq_file(file,l_file):

    train_file = np.load(file)
    train_file = np.array(train_file,dtype=np.uint8)
    label_file = np.load(l_file)
    label_file = np.array(label_file,dtype=np.long)
    return train_file,label_file

