# -*- encoding: utf-8 -*-
'''
-----------------------------------
    File name:   utils.py
    Time:        2022/09/22 14:55:08
    Editor:      ZhiyuGeGe & Figo
-----------------------------------
'''
import torch
import numpy as np


def get_mask_matrix(activate: 'np.ndarray', radii1: 'int', radii2: 'int', order='l1'):

    """ ----- Expect the energy shape is (H x W) ------ """

    height, width = activate.shape
    
    """get the centralization"""
    crow, ccol = int(height / 2), int(width / 2)
    centralization = [crow, ccol]
    mask_matrix1, mask_matrix2 = np.zeros((height, width), dtype=np.int64), np.zeros((height, width), dtype=np.int64)
    xaxis, yaxis = np.ogrid[:height, :width]

    if order == "l1":
        mask_area = abs(xaxis - centralization[0]) + abs(yaxis - centralization[1]) >= radii1
        mask_matrix1[mask_area] = 1
        mask_area = abs(xaxis - centralization[0]) + abs(yaxis - centralization[1]) <= radii2
        mask_matrix2[mask_area] = 1
    
    if order == "l2":
        mask_area = abs(xaxis - centralization[0])**2 + abs(yaxis - centralization[1])**2 >= radii1**2
        mask_matrix1[mask_area] = 1
        mask_area = abs(xaxis - centralization[0])**2 + abs(yaxis - centralization[1])**2 <= radii2**2
        mask_matrix2[mask_area] = 1
    
    if order == 'square':
        mask_matrix1[crow-radii1: crow+radii1, ccol-radii1: ccol+radii1] = 1
        mask_matrix2[crow-radii2: crow+radii2, ccol-radii2: ccol+radii2] = 1
        mask_area = mask_matrix2 - mask_matrix1
        return mask_area
    
    mask = mask_matrix1 + mask_matrix2 - 1
    mask[crow:, :] = 0
    mask = mask + np.flipud(mask)
    mask[:, ccol:] = 0
    mask = mask + np.fliplr(mask) 
    return mask


def min_max_normalize(x):
    return (x-x.min()) / (x.max()-x.min())


def torch_freq_mask_(data, mask=None, is_mulit_channels=True):
    """ ---------- Mask the image in frequency domain (suit for torch)---------- """
    if is_mulit_channels is True:
        f1 = torch_freq_mask_(data[0, :, :], mask, is_mulit_channels=False).unsqueeze(dim=0)
        f2 = torch_freq_mask_(data[1, :, :], mask, is_mulit_channels=False).unsqueeze(dim=0)
        f3 = torch_freq_mask_(data[2, :, :], mask, is_mulit_channels=False).unsqueeze(dim=0)
        return torch.cat([f1, f2, f3], dim=0)
    """ ----------- Get freq image -----------"""
    freq_image = torch.fft.fft2(data)  # Turn to frequency domain
    freq_image = torch.fft.fftshift(freq_image) * mask  # mask the desired area
    return freq_image


def torch_freq_mask(img_batch: torch.Tensor, mask: torch.Tensor):
    """
    Mask the image from an image batch in frequency domain (suit for torch)
    Args:
        :: img_batch: Image batch with shape (B x 3 x H x W) (torch.float32)
        :: mask: Mask matrix with shape (H x W) 
    Return:
        :: img_batch: Mask image batch with shape (B x 3 x H x W)  (torch.complex32)
    """
    new_image = []
    is_mulit_channels = True if img_batch.shape[1] == 3 else False
    for image in img_batch:
        mask_image = torch_freq_mask_(image, mask, is_mulit_channels)
        new_image.append(mask_image.unsqueeze(dim=0))
    return torch.cat(new_image, dim=0)


def freq_restructuring(lf_img: torch.Tensor, hf_img: torch.Tensor, radii: int):
    """
    Args: 
        :: lf_img: The low frequency retain image
        :: hf_img: The high frequency retain image
        :: radii: The radii of mask matrix
    Return:
        :: the restructuring frequency image, shape with (B x 3 x H x W)
    """
    assert lf_img.shape == hf_img.shape
    B, C, H, W = lf_img.shape
    is_mulit_channels = False if C == 1 else True
    mask_lf = get_mask_matrix(np.zeros((H, W)), radii1=0, radii2=radii, order='l2')
    mask_hf = get_mask_matrix(np.zeros((H, W)), radii1=radii+1, radii2=H, order='l2')
    restructuring_ = torch_freq_mask(lf_img, mask_lf) + torch_freq_mask(hf_img, mask_hf)
    new_image = []
    """ --------- restructuring the frequency image --------- """
    for channel in range(C):
        new_channel_image = []
        for batch_index in range(B):
            freq_image = restructuring_[batch_index, channel, :, :]
            freq_image = torch.fft.ifftshift(freq_image)
            freq_image = torch.fft.ifft2(freq_image)                                        # Turn to spatial domain
            freq_view = torch.real(freq_image)
            freq_view = min_max_normalize(freq_view)                                        # normalization is quit important!
            new_channel_image.append(freq_view.unsqueeze(dim=0))                            # (B x H x W)
        new_image.append(torch.cat(new_channel_image, dim=0).unsqueeze(dim=1))              # (B x 1 x H x W)
    return torch.cat(new_image, dim=1)                                                      # (B x C x H x W)


def visual_spectral_density(data: torch.Tensor):
    """
    Args:
        :: data: image batch with shape: (B, C, H, W)
    Return:
        :: the density of each image
    """
    psd1D_img = []
    for _, image in enumerate(data):
        image_numpy = image.permute(1, 2, 0)
        img_gray = RGB2gray(image_numpy) if data.shape[1] == 3 else image_numpy[:, :, 0]
        freq_image = torch.fft.fft2(img_gray)
        freq_image_shift = torch.fft.fftshift(freq_image)
        magnitude_spectrum = 20 * torch.log(1+torch.abs(freq_image_shift))
        psd1D = azimuthalAverage(magnitude_spectrum, None)
        psd1D = (psd1D - psd1D.min()) / (psd1D.max() - psd1D.min())
        psd1D_img.append(psd1D.unsqueeze(dim=0))
    return torch.cat(psd1D_img)


def RGB2gray(rgb):
    """Turn RGB image to Gray image"""
    if rgb.shape[-1] != 3:
        raise ValueError("Expected RGB image, but only got {} channel".format(rgb.shape[-1]))
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    return 0.2989 * r + 0.5870 * g + 0.1140 * b


def normaldensity(mu, sigma, data):
    """get the density of normal distribution"""
    N1 = np.sqrt(2 * np.pi * np.power(sigma, 2))
    fac1 = np.power(data - mu, 2) / np.power(sigma, 2)
    return np.exp(-fac1 / 2) / N1


def doubledensity(Heights: int, data: np.ndarray, rho: float):
    sigma1, sigma2, mu1, mu2 = Heights / 10, Heights / 10, 0, 16
    density = normaldensity(mu1, sigma1, data) + rho*normaldensity(mu2, sigma2, data)
    return (density - density.min()) / (density.max() - density.min())


def azimuthalAverage(image, center=None):
    """
    Calculate the azimuthally averaged radial profile.
    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is 
             None, which then uses the center of the image (including 
             fracitonal pixels).
    
    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    if not center:
        """ get the center (default the center of an image)"""
        center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])

    r = np.hypot(x - center[0], y - center[1])

    # Get sorted radii
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flatten()[ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.astype(int)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = np.where(deltar)[0]       # location of changed radius
    nr = rind[1:] - rind[:-1]        # number of radius bin
    
    # Cumulative sum to figure out sums for each radius bin
    csim = torch.cumsum(i_sorted, dim=0, dtype=torch.float32)
    tbin = csim[rind[1:]] - csim[rind[:-1]]
    radial_prof = tbin / torch.tensor(nr).cuda()

    return radial_prof


def get_toy_restructuring_image(target, radii, model, dataset):
    """ 
    Args:
        :: target: the target class
        :: noise: (bool) add the random noise or not
        :: radii: the radii of restructuring
    Return: 
        The toy image which is restructured by toy model (frequency restructuring image)
    """
    from tqdm import tqdm

    if target is None:
        loader = dataset.make_loaders(batch_size=100, workers=4, is_shuffle=True)
    else:
        loader = dataset.make_class_loader(target, batch_size=100, workers=4)                       # load image from target class
    iterator = tqdm(enumerate(loader), total=len(loader))
    toy_image, Ori_img = [], []
    print(f"=> Class {target} dataLoader is OK!")
    
    model.eval()
    for i, (inp, _) in iterator:
        noise = min_max_normalize(torch.rand_like(inp, dtype=torch.float32)) * inp.mean() / 10      # adding noise (which helps change the spectral density)
        real_image = inp + noise                                                                    # real input image + random noise
        fake_image = model(real_image.cuda())
        toy_image.append(fake_image.cpu().detach())
        Ori_img.append(inp.cpu().detach())
    new_img, ori_img = torch.cat(toy_image, dim=0), torch.cat(Ori_img, dim=0)
    restructuring_img = freq_restructuring(ori_img, new_img, radii)                                 # restructure image
    return restructuring_img
