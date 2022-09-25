# -*- encoding: utf-8 -*-
'''
-----------------------------------
    File name:   generate_toy_dataset.py
    Time:        2022/09/22 15:23:01
    Editor:      ZhiyuGeGe & Figo
-----------------------------------
'''

import os
import torch
import torchvision as tv
from argparse import ArgumentParser
from robustness import defaults
from config import OTHER_ARGS, MODEL_DATA_ARGS
from utils_data import cifar10
from toy_dataset import toy_kernel
from utils import get_toy_restructuring_image

"""
    python generate_toy_dataset.py --exp_name WCR --cuda_index 0
"""

parser = ArgumentParser()
parser = defaults.add_args_to_parser(OTHER_ARGS, parser)                                            # cuda_index exp_name
parser = defaults.add_args_to_parser(MODEL_DATA_ARGS, parser)                                       # arch, batch_size, workers
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = "{}".format(args.cuda_index)

""" ------------ Import the dataloader ------------- """
transforms = tv.transforms.ToTensor()
dataset_path = "{}-Toy-Dataset".format(args.exp_name)
if not os.path.exists(dataset_path): 
    os.mkdir(dataset_path)

for loop_type in ['train', 'test']:
    is_train_loop = True if loop_type == 'train' else False
    ds = cifar10(data_path="your_cifar10_path", is_train=is_train_loop, transform=transforms)
    folder_path = os.path.join(dataset_path, "{}/".format(loop_type))
    if not os.path.exists(folder_path): os.mkdir(folder_path)
    print("=> The exp is done in path:{}".format(folder_path))

    if args.exp_name == "WCR":
        # weak class related model
        model_path = "WCR/all_class/Epoch0.pth"
        model = toy_kernel(number=16, kernel_size=7, channels=3)
        model_dicts = torch.load(model_path)['model']
        model = torch.nn.DataParallel(model).cuda()
        model.load_state_dict(model_dicts)
        print("=> Model load all parameters!")
        
        image = get_toy_restructuring_image(None, 10, model, ds)
        print("=> The image shape is:{}".format(image.shape))
        torch.save(image, os.path.join(folder_path, "all.pth".format()))
    
    elif args.exp_name == "SCR":
        for target in range(10):
            # strong class related model
            model = toy_kernel(number=16, kernel_size=7, channels=3)
            model = torch.nn.DataParallel(model).cuda()
            model_path = "SCR/class{}/Epoch0.pth".format(target)
            model_dicts = torch.load(model_path)['model']
            model.load_state_dict(model_dicts)
            print("=> Model load all parameters!")

            image = get_toy_restructuring_image(target, 10, model, ds)
            print("=> The image shape is:{}".format(image.shape))
            torch.save(image, os.path.join(folder_path, "{}.pth".format(target)))
