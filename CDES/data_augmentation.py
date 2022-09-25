# -*- encoding: utf-8 -*-
'''
-----------------------------------
    File name:   data_augmentation.py
    Time:        2021/12/12 12:19:58
    Editor:      ZhiyuGeGe & Figo
-----------------------------------
'''
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as tvF

from utils import min_max_normalize, get_mask_matrix, torch_freq_mask_
from utils_data import *
from toy_dataset import toy_kernel


class RandomGaussianBlur(transforms.GaussianBlur):
    def __init__(self, prob, kernel_size, sigma=(0.1, 2.0)):
        super(RandomGaussianBlur, self).__init__(kernel_size, sigma)
        self.prob = prob
    
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if self.prob <= np.random.uniform(0, 1):
            sigma = self.get_params(self.sigma[0], self.sigma[1])
            bulr_img = tvF.gaussian_blur(img, self.kernel_size, [sigma, sigma])
            return min_max_normalize(bulr_img)
        else: return img
    
    def __repr__(self):
        s = '(kernel_size={}, '.format(self.kernel_size)
        s += 'sigma={})'.format(self.sigma)
        return self.__class__.__name__ + s


class MinMaxNormalization(nn.Module):
    def __init__(self):
        super(MinMaxNormalization, self).__init__()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return min_max_normalize(tensor)

    def __repr__(self):
        return self.__class__.__name__


class EnergyMixup(nn.Module):
    def __init__(self, target, prob):
        super(EnergyMixup, self).__init__()
        model_path = "your_path".format(target)
        model_dicts = torch.load(model_path)['model']  # load the parameters of toy model
        model = toy_kernel(number=8, kernel_size=5, channels=3)
        self.model = torch.nn.DataParallel(model).cuda()
        self.model.load_state_dict(model_dicts)
        self.prob = prob
    
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        C, H, W = tensor.shape
        mask_lf = get_mask_matrix(np.zeros((H, W)), radii1=0, radii2=10, order='l2')
        mask_hf = get_mask_matrix(np.zeros((H, W)), radii1=11, radii2=H, order='l2')
        if self.prob <= np.random.uniform(0, 1):
            noise = min_max_normalize(torch.rand_like(tensor, dtype=torch.float32)) * tensor.mean() / 10
            real_image = (tensor + noise).unsqueeze(dim=0).type(torch.FloatTensor)
            fake_image = self.model(real_image.cuda()).squeeze(dim=0).cpu()
            restructuring_ = torch_freq_mask_(tensor.cpu().detach(), mask_lf) + torch_freq_mask_(fake_image.cpu().detach(), mask_hf)
            new_channel_image = []
            for channel in range(C):
                freq_image = torch.fft.ifftshift(restructuring_[channel, :, :])
                freq_image = torch.fft.ifft2(freq_image)  # Turn to spatial domain
                freq_view = torch.real(freq_image)
                freq_view = min_max_normalize(freq_view)  # normalization is quit important!
                new_channel_image.append(freq_view.unsqueeze(dim=0))
            return torch.cat(new_channel_image, dim=0)  # (C x H x W)
        else: return tensor
    
    def __repr__(self):
        return self.__class__.__name__