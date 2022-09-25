# -*- encoding: utf-8 -*-
'''
-----------------------------------
    File name:   train_toy_models.py
    Time:        2022/09/22 15:22:28
    Editor:      ZhiyuGeGe & Figo
-----------------------------------
'''
import os
import cox
import numpy as np
import torchvision as tv
from cox.utils import Parameters
from robustness import defaults
from argparse import ArgumentParser
from torch.utils.data import ConcatDataset, DataLoader

from utils import doubledensity
from utils_data import cifar10
from toy_dataset import toy_kernel, make_toy_data_kernel
from config import OTHER_ARGS, UPDATE_ARGS, MODEL_DATA_ARGS

"""
    python train_toy_models.py --number 16 --kernel_size 7 --cuda_index 0 --weight_decay 1e-4 --momentum 0.9 --epochs 20 --lr 0.01 --step_lr 5 --step_lr_gamma 0.1 --exp_name SCR --batch_size 100 --workers 4
"""

parser = ArgumentParser()
Args = [['number', int, "The number of kernels", 3], ['kernel_size', int, "The shape of kernel size", 3]]
parser = defaults.add_args_to_parser(Args, parser)
parser = defaults.add_args_to_parser(OTHER_ARGS, parser)                            # cuda_index exp_name
parser = defaults.add_args_to_parser(UPDATE_ARGS, parser)                           # epochs, lr, momentum, weight_decay, step_lr, step_lr_gamma
parser = defaults.add_args_to_parser(MODEL_DATA_ARGS, parser)                       # arch, batch_size, workers
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = "{}".format(args.cuda_index)

""" ------------ Import the Model ------------ """
model = toy_kernel(number=args.number, kernel_size=args.kernel_size, channels=3).cuda()
print(f"=> toy model is OK!")

""" ------------ Import the dataloader (strong class related) ------------- """
transforms = tv.transforms.ToTensor()
ds_train = cifar10(data_path="your_cifar10_path", is_train=True, transform=transforms)
ds_test = cifar10(data_path="your_cifar10_path", is_train=False, transform=transforms)
print("=> Dataset is OK...")

image_density = doubledensity(Heights=32, data=np.arange(0, 20, 1), rho=1.0)  # 获取double density
print("=> Density criteria is OK!")

saving_path = args.exp_name
if not os.path.exists(saving_path):
    os.mkdir(saving_path)

if args.exp_name == "SCR":
    print("=> strong class related training...")
    for target_label in range(10):
        target_saving_path = os.path.join(saving_path, "class{}/".format(target_label))
        print(f"=> Target saving path is:{target_saving_path}")
        
        train_kwargs = {
            "epochs": args.epochs, 
            "lr": args.lr, 
            "momentum": args.momentum, 
            'out_dir': target_saving_path,
            "weight_decay": args.weight_decay, 
            "step_lr": args.step_lr, 
            "step_lr_gamma": args.step_lr_gamma
        }
        
        train_args = Parameters(train_kwargs)
        train_args = defaults.check_and_fill_args(train_args, UPDATE_ARGS, None)
        out_store = cox.store.Store(saving_path, exp_id="class{}/".format(target_label))
        print(f"=> The exp path is:{target_saving_path}")

        dataset = ConcatDataset([ds_train.make_class(target=target_label), ds_test.make_class(target=target_label)])
        loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
        print(f"=> Got {len(loader.dataset)} images in dataloader")
        
        make_toy_data_kernel(train_args, model, loader, image_density, out_store)
        print("="*40, "Finished class {} training".format(target_label), "="*40)


if args.exp_name == "WCR":
    print("=> weak class related training...")
    
    dataset = ConcatDataset([ds_train.make_datasets(), ds_test.make_datasets()])
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
    print("=> Got {} images in dataloader".format(len(loader.dataset)))

    target_saving_path = os.path.join(saving_path, "all_class/")
    train_kwargs = {
        "epochs": args.epochs, 
        "lr": args.lr, 
        "momentum": args.momentum, 
        'out_dir': target_saving_path, 
        "weight_decay": args.weight_decay, 
        "step_lr": args.step_lr, 
        "step_lr_gamma": args.step_lr_gamma
    }
    train_args = Parameters(train_kwargs)
    train_args = defaults.check_and_fill_args(train_args, UPDATE_ARGS, None)
    out_store = cox.store.Store(saving_path, exp_id="all_class/")
    print(f"=> The exp path is:{target_saving_path}")

    make_toy_data_kernel(train_args, model, loader, image_density, out_store)
    print("="*40, "Finished datasets training", "="*40)