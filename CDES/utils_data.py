# -*- encoding: utf-8 -*-
'''
-----------------------------------
    File name:   utils_data.py
    Time:        2022/09/22 19:57:47
    Editor:      ZhiyuGeGe & Figo
-----------------------------------
'''

import os
import torch
import warnings
import random as rd
import torch.nn as nn
from utils import *
from data_augmentation import *
from typing import Tuple
from robustness.tools import folder
from robustness.data_augmentation import *
from robustness.datasets import *
from robustness.tools.folder import *
from robustness.tools import folder, helpers
from robustness.attacker import Attacker
from torch.utils.data import Dataset, DataLoader


def make_adv_image(model, loader, eps, iters, is_target=False, with_outs=False):
    """ 
    Make the Adversarial Examples (based on the robustness module, only suit for spectial model and dataset)
    Args:
        :: model: the model which has loaded the parameters
        :: loader: the image loader to build AEs
        :: eps: (float) the epsilon of perturbation
        :: iters: (int) the attack steps
        :: is_target: (bool) the targeted attack or not
    Return:
        :: the AEs and its' predictions
    """
    from tqdm import tqdm
    pgd_attack_kwargs = {
        'constraint': '2', # L2 fgsm
        'eps': eps, # Epsilon constraint (L2 norm)
        'step_size': float(eps) * 2/3, # Learning rate for PGD
        'iterations': iters, # 1, 2, 3
        'targeted': is_target, # Istargeted attack
    }
    model.eval()
    Adv_image, Adv_label, Adv_out, score = [], [], [], 0
    itereator = tqdm(enumerate(loader), total=len(loader))
    for _, (image, label) in itereator:
        adv_out, adv_im = model(image.cuda(), label.cuda(), make_adv=True, **pgd_attack_kwargs)
        adv_pre_label = torch.max(adv_out, dim=1)[1]
        score += (adv_pre_label != label.cuda()).sum().item()
        Adv_image.append(adv_im.cpu().detach())  # adversarial examples
        Adv_label.append(adv_pre_label.cpu().detach())  # the predict label of AEs
        Adv_out.append(adv_out.cpu().detach())
    print("==> Adv successful rate is:{:.3f}".format(score/(len(loader)*loader.batch_size)))
    
    if with_outs: return torch.cat(Adv_image, dim=0), torch.cat(Adv_label, dim=0), torch.cat(Adv_out, dim=0)
    
    return torch.cat(Adv_image, dim=0), torch.cat(Adv_label, dim=0)


class attack(nn.Module):
    def __init__(self, model, dataset):
        """ Make the Adversarial Examples (based on the robustness module, only suit for spectial model and dataset) """
        super(attack, self).__init__()
        self.model = model
        self.normalizer = helpers.InputNormalize(dataset.mean.cuda(), dataset.std.cuda())
        self.attacker = Attacker(model, dataset)
    
    def forward(self, inp, target, with_latent=False, **attacker_kwargs):
        """
        Args:
            :: inp: (torch.Tensor) The input tensor (B x C x H x W)
            :: target: (torch.Tensor) The target label
            :: constraint: (str) The adv constraint (l2, linf)
            :: step_size: (float) The step size for adversarial attacks (alpha)
            :: iterations: (int) The number of steps for PGD attack
            :: eps: (float) The epsilon of attacks
            :: targeted: (bool): if True (False), minimize (maximize) the loss
        Return:
            :: adversarial examples
        """
        self.eval()
        adv_image = self.attacker(inp, target, **attacker_kwargs)
        if with_latent:
            adv_image = self.normalizer(adv_image)
            outs, layers, Feature = self.model(adv_image, with_latent=True)
            return adv_image, (outs, layers, Feature)
        else: return adv_image


def is_path_exists(path):
    """ Check if the path does not exists!"""
    if not os.path.exists(path):
        raise ValueError("No such file or folder exists, check the path {} please!".format(path))
    else: return path


def read_from_npy(path, name):
    path = is_path_exists(path)
    file_path = is_path_exists(os.path.join(path, name))
    data = np.load(file_path)
    label = np.array([1]*data.shape[0], dtype=np.long)
    return data, label


def read_from_pth(path, name):
    path = is_path_exists(path)
    file_path = is_path_exists(os.path.join(path, name))
    data = torch.load(file_path)
    label = torch.tensor(np.array([1]*data.shape[0]), dtype=torch.long)
    return data.detach(), label


class BasicDataset(object):
    
    def __init__(self, data_path, image_type='rgb', is_norm=False):
        """
        Args:
            :: data_path: the path of numeric data files
            :: image_type: the type of image ('rgb', 'gray')
            :: is_norm: (bool) data augmentation, data normalization
        Returns:
            :: classes: the class of dataset
            :: num_classes: the number of classes in the dataset
            :: file_type: the type of numeric data files ('pth', 'npy')
            :: data: the data
            :: label: the label
            :: mean: the mean of data
            :: std: the std of data
        """
        super(BasicDataset, self).__init__()
        self.path = is_path_exists(data_path)
        self.image_type, self.is_norm = image_type, is_norm
        self.classes, self.num_classes, self.file_type = self._find_class(self.path)
        self.data, self.label = self._read_from_files()
        self._assert_negative()
        self.Heights = self.data.shape[2]
        self.length = self.__len__()
        
        if self.data.shape[1] == 3 and image_type == 'gray':
            self.data = self.rgb2gray(self.data)
        elif self.data.shape[1] == 1 and image_type == 'rgb':
            raise TypeError("Got RGB type, but only {} channels".format(self.data.shape[1]))
        self.mean, self.std = self._get_mean_std()
        if self.is_norm: self.data = self.normalization(self.data, self.mean, self.std)
    
    def _find_class(self, path):
        """
        Find the class number of a dataset
        
        Returns:
            tuple: (classes, number) where classes are relative to (dir)
        """
        name_list = [name.split('.')[0] for name in os.listdir(path)]
        return np.sort(name_list), len(name_list), os.listdir(path)[0].split('.')[1]
    
    def _assert_negative(self):
        if (self.data < 0.0).sum().item() > 0:
            warnings.warn("The image matrix should range in (0, 1.0)")

    def normalization(self, data, mean, std):
        if self.file_type == 'pth':
            norm_ = torch.zeros_like(data)
            for idx, img in enumerate(data):
                if self.image_type == "rgb":
                    for c in range(data.shape[1]): norm_[idx, c, :, :] = (data[idx, c, :, :] - mean[c]) / std[c]
                else: norm_[idx, :, :, :] = (data[idx, :, :, :] - mean) / std
        
        elif self.file_type == 'npy':
            norm_ = np.zeros_like(data)
            for idx, img in enumerate(data):
                if self.image_type == "rgb":
                    for c in range(data.shape[1]): norm_[idx, c, :, :] = (data[idx, c, :, :] - mean[c]) / std[c]
                else: norm_[idx, :, :, :] = (data[idx, :, :, :] - mean) / std
        print("The data has been normalization!")
        return norm_
    
    def _read_from_files(self):
        Data, Label, file_name = [], [], np.sort(os.listdir(self.path))
  
        if self.file_type == "pth":
            for idx, name in enumerate(file_name):
                data, label = read_from_pth(path=self.path, name=name)
                Data.append(data), Label.append(label*idx)
            return torch.cat(Data, dim=0), torch.cat(Label, dim=0)
        
        elif self.file_type == "npy":
            for idx, name in enumerate(file_name):
                data, label = read_from_npy(path=self.path, name=name)
                Data.append(data), Label.append(label*idx)
            return np.concatenate(Data, axis=0), np.concatenate(Label, axis=0)
    
    def _get_mean_std(self):
        if self.image_type == "rgb":
            mean = [self.data[:, c, :, :].mean().item() for c in range(3)]
            std = [self.data[:, c, :, :].std().item() for c in range(3)]
        
        elif self.image_type == "gray":
            mean, std = self.data[:, 0, :, :].mean(), self.data[:, 0, :, :].std()
        return torch.tensor(mean), torch.tensor(std)
    
    def rgb2gray(self, data):
        if self.file_type == "pth":
            new_data = torch.zeros((data.shape[0], 1, data.shape[2], data.shape[3]), dtype=torch.float32)
            for idx, img in enumerate(data):
                new_data[idx, :, :, :] = RGB2gray(img.permute(1, 2, 0)).unsqueeze(dim=-1).permute(2, 0, 1)
        
        elif self.file_type == 'npy':
            new_data = np.zeros((data.shape[0], 1, data.shape[2], data.shape[3]), dtype=np.float32)
            for idx, img in enumerate(data):
                new_data[idx, :, :, :] = RGB2gray(img.trnaspose(1, 2, 0)).expand_dims(axis=-1).trnaspose(2, 0, 1)
        print("Turn the RGB image to Gray image! The new data shape is: {}".format(new_data.shape))
        return new_data
    
    def __len__(self):
        return self.data.shape[0]


class TensorDataAugDataset(Dataset[Tuple[torch.Tensor, ...]]):
    """ The data augumentation of tensor data """
    def __init__(self, *tensors: torch.Tensor, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        assert transform is not None
        self.tensors = tensors
        self.transform = transform
    
    def __getitem__(self, index):
        return self.transform(self.tensors[0][index]), self.tensors[1][index]

    def __len__(self):
        return self.tensors[0].size(0)


class Path2Image(torch.utils.data.Dataset):
    """ Path to pil image, then to tensor
    Args:
        :: path_list: tuple (image path, target)
        :: loader: the way of reading image
        :: transform: the data augmentation
    Return:
        :: dataset
    """
    def __init__(self, path_list, loader, transform=None):
        super(Path2Image, self).__init__()
        self.path_list = path_list
        if transform is None:
            raise ValueError("The transform is None!")
        else: self.transform = transform
        self.loader = loader
        self.label = [item[1] for item in path_list]
        self.path = [item[0] for item in path_list]
        self.length = self.__len__()
    
    def __getitem__(self, index):
        path, target = self.path_list[index]
        sample = self.transform(self.loader(path))
        return sample, target

    def __len__(self):
        return len(self.path_list)


class dataset2class(object):
    def __init__(self, data_path, size, is_train=False, image_type='rgb', transform=None, **kwargs):
        """
        Args:
            :: data_path: The image path, which contains (train, test) image folder
            :: size: the size of shape for transform.radnomcrop
            :: is_train: the train folder or test folder
        
        Kwargs:
            :: num_classes: the number of classes
            :: mean: the mean of the dataset
            :: std: the std of the dataset
            :: cumstom_class: the name of each class in the dataset
        
        Return:
            :: classes: The classes of the dataset
            :: num_classes: The number of classes
            :: transform: Data Augmentation defaults
            :: path: the data path 
            :: dataset: the dataset
            :: length: the number of image if dataset
        """
        super(dataset2class, self).__init__()
        self.size, self.image_type = size, image_type
        if transform is None:
            self.transform = TRAIN_TRANSFORMS_DEFAULT(self.size) if is_train else TEST_TRANSFORMS_DEFAULT(self.size)
        else:
            self.transform = transform
        print("=> Transform is:{}".format(self.transform))

        """ ----------- check whether the path exist ----------- """
        path = os.path.join(data_path, "train/" if is_train else "test")
        self.path = is_path_exists(path)
        self.classes, self.num_classes = self._find_class(self.path)
        self.classes_dicts = {idx: name for idx, name in enumerate(self.classes)}
        self.loader = pil_loader_gray if image_type == "gray" else pil_loader
        self.dataset = folder.ImageFolder(root=self.path, transform=self.transform, loader_type=self.image_type)
        self.length = self.dataset.__len__()
        self.__dict__.update(kwargs)
    
    def _find_class(self, path):
        """ Find the class number of a dataset
        Returns:
            tuple: (classes, number) where classes are relative to (dir)
        """
        name_list = [int(name.split('.')[0]) for name in os.listdir(path)]
        return np.sort(name_list), len(name_list)
    
    def _check_path(self, path):
        """ Check the image list, if None, check the path or your code
        """
        if len(path) == 0: 
            warnings.warn("Got 0 image path, please check the path or your code!")
    
    def make_datasets(self):
        """
        Make the datasets, which can build a total dataloader(contains all classes!)
        """
        return Path2Image(self.dataset.samples, self.loader, self.transform)

    def make_loaders(self, batch_size, workers, is_shuffle=True):
        """
        Make the dataloaders, which can build a total dataloader(contains all classes!)
        """
        dataset = Path2Image(self.dataset.samples, self.loader, self.transform)
        return DataLoader(dataset, batch_size=batch_size, num_workers=workers, pin_memory=True, shuffle=is_shuffle)
    
    def make_class(self, target):
        """ Make the dataset of target class
        Args: 
            :: target: the target class
        Return:
            :: the dataset
        """
        class_name = self.classes_dicts[target]
        if class_name not in self.custom_class:
            raise ValueError("target {} is not in the current dataset!".format(class_name))
        target_path = [(path, label) for path, label in self.dataset.samples if int(label) == int(target)]
        self._check_path(target_path)
        return Path2Image(target_path, self.loader, self.transform)
        
    def make_class_loader(self, target, batch_size, workers):
        dataset = self.make_class(target)
        return DataLoader(dataset, batch_size=batch_size, pin_memory=True, num_workers=workers)

    def make_mini(self, mini):
        """ Make a mini dataset
        Args: 
            :: mini: the number of images in the mini datasets (
                if float: the sample rate,
                if int: the number of sampling 
                )
        Return:
            :: the dataset
        """
        if isinstance(mini, int):
            assert  1 <= mini <= self.length
        if isinstance(mini, float):
            assert 0 <= mini <= 1
            mini = int(mini * self.length)
        path_list = self.dataset.samples
        rd.shuffle(path_list)
        sample_path = path_list[0: mini]
        self._check_path(sample_path)
        return Path2Image(sample_path, self.loader, self.transform)
  
    def make_mini_target_loader(self, mini, target, batch_size):
        """ Make a mini target dataset
        Args:
            :: mini: the number of images in the mini datasets
            :: target: the target class
        Return:
            dataset
        """
        class_name = self.classes_dicts[target]
        print("=> Got the class {}".format(class_name))
        if class_name not in self.custom_class:
            raise ValueError("target {} is not in the current dataset!".format(class_name))
        
        path_list, target_path, index = self.dataset.samples, [], 1
        rd.shuffle(path_list)
        for idx, (path, label) in enumerate(path_list):
            if label == target:
                target_path.append((path, label))
                index += 1
            elif index <= mini: continue
            else: break
    

        self._check_path(target_path)
        dataset = Path2Image(target_path, self.loader, self.transform)
        return DataLoader(dataset, batch_size=batch_size, num_workers=0, pin_memory=True)


def get_mean_std(dataset, channels):
    mean, std = [0]*channels, [0]*channels
    for c in range(channels):
        data = []
        for image, label in dataset: data.append(image[c, :, :])
        mean[c] = torch.cat(data, dim=0).mean().detach().numpy().item()
        std[c] = torch.cat(data, dim=0).std().detach().numpy().item()
    return mean, std


class cifar10(dataset2class):
    def __init__(self, data_path, is_train=False, transform=None):
        ds_kwargs = {
            'num_classes': 10,
            'mean': torch.tensor([0.4914, 0.4822, 0.4465]),
            'std': torch.tensor([0.2023, 0.1994, 0.2010]),
            'custom_class': [_ for _ in range(10)],  # ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
        }
        super(cifar10, self).__init__(data_path, 32, is_train, transform=transform, **ds_kwargs)


class mnist(dataset2class):
    def __init__(self, data_path, is_train=False, transform=None):
        ds_kwargs = {
            'num_classes': 10,
            'mean': torch.tensor([0.1307,]),
            'std': torch.tensor([0.3081,]),
            'custom_class': [_ for _ in range(10)],
        }
        super().__init__(data_path, 28, is_train, "gray", transform, **ds_kwargs)


class svhn(dataset2class):
    def __init__(self, data_path, is_train=False, transform=None):
        ds_kwargs = {
            'num_classes': 10,
            'mean': torch.tensor([0.485,0.456,0.406]),
            'std': torch.tensor([0.229,0.224,0.225]),
            'custom_class': [_ for _ in range(10)],
        }
        super().__init__(data_path, 32, is_train, "rgb", transform, **ds_kwargs)


class imagenet(dataset2class):
    def __init__(self, data_path, is_train=False, transform=None):
        ds_kwargs = {
            'num_classes': 1000,
            'mean': torch.tensor([0.485, 0.456, 0.406]),
            'std': torch.tensor([0.229, 0.224, 0.225]),
            'custom_class': [_ for _ in range(1000)],
        }
        super().__init__(data_path, 224, is_train, "rgb", transform, **ds_kwargs)
