# -*- encoding: utf-8 -*-
'''
-----------------------------------
    File name:   config.py
    Time:        2022/09/22 20:29:15
    Editor:      ZhiyuGeGe & Figo
-----------------------------------
'''

import torchvision as tv

""" ---------- mask experiment ---------- """
OTHER_ARGS = [
    ['exp_name', str, 'The name of your experiment', "Exp"],
    ['cuda_index', int, 'The index of GPU', 0],
]

UPDATE_ARGS = [
    ['epochs', int, "The epoch of training stage", 1],
    ['random_drop', int, "Random drop the input frequency", 1],
    ['target', int, "The targer class of dataset", 0],
    ['hard_mask', bool, "The hard mask matrix (0-1)", True],
    ['lr', float, "The learning rate of updating the mask matrix", 0.1],
    ['momentum', float, "The momentum of optimizer SGD", 0.9],
    ['weight_decay', float, "The weight decay of optimizer SGD", 1e-4],
    ['step_lr', int, "The step lr of schedule", 80],
    ['step_lr_gamma', float, "The step lr gamma of schedule", 0.1],
    ['freq_test', int, "Beigin the frequency test", 0],
    ['out_dir', str, "The saving path", None],
    ['fusion', bool, "The frequency fusion", False],
    ['gradient_test', bool, "The gradient test for each val stage", False]
]

ADV_ARGS = [
    ['constraint', str, "The constraint of adversarial perturbations", '2'],
    ['eps', float, "The epsilon adversarial perturbations", 0.5],
    ['step_size', float, "The step size of adversarial perturbations", 0],
    ['iters', int, "The itereations of adversarial perturbations", 1],
    ['is_target', bool, "The target attack or not", False],
]

MODEL_DATA_ARGS = [
    ['arch', str, "The Moddel'name", "vgg11"],
    ['is_resume', bool, "Load the pretrain model paramters", False],
    ['batch_size', int, "The batch size of datasets", 50],
    ['workers', int, "The workers on one process", 4],
]

TENSOR_TRAIN_TRANSFORM_DEFAULT = lambda size: tv.transforms.Compose([
    tv.transforms.ToPILImage(),
    tv.transforms.RandomCrop(size, padding=4),
    tv.transforms.RandomHorizontalFlip(),
    tv.transforms.ColorJitter(.25,.25,.25),
    tv.transforms.RandomRotation(2),
    tv.transforms.ToTensor(),
])

TENSOR_TEST_TRANSFORMS_DEFAULT = lambda size: tv.transforms.Compose([
    tv.transforms.ToPILImage(),
    tv.transforms.Resize(size),
    tv.transforms.CenterCrop(size),
    tv.transforms.ToTensor()
])


TRANSFORM_DEFAULT = tv.transforms.Compose([
    tv.transforms.ToTensor(),
])