#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import random

def normal_img(img):
    mx = img.max()
    mn = img.min()
    img = (img-mn)/(mx-mn)
    return img

def fft(img):
    return np.fft.fft2(img)

def fftshift(img):
    return np.fft.fftshift(fft(img))

def distance(i, j, imageSize, r):
    dis = np.sqrt((i - imageSize/2) ** 2 + (j - imageSize/2) ** 2)
    if dis < r:
        return 1.0
    else:
        return 0

def mask_radial(r):
    rows, cols = 32,32
    mask = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            mask[i, j] = distance(i, j, imageSize=rows, r=r)
    return mask
# DFT
def generate_different_dft(Images, r):
    low_mask = mask_radial(r)
    high_mask = 1-low_mask
    Images_freq_low = []
    Images_freq_high = []
    for i in tqdm(range(Images.shape[0])):
        img = Images[i]
        tmp_low = np.zeros_like(img,dtype=complex)
        tmp_high = np.zeros_like(img,dtype=complex)
        for j in range(3):
            dft_im = fftshift(np.float32(img[:,:,j]))
            tmp_low[:,:,j] = dft_im*low_mask
            tmp_high[:,:,j] = dft_im*high_mask
        Images_freq_low.append(tmp_low)
        Images_freq_high.append(tmp_high)
    return np.array(Images_freq_low), np.array(Images_freq_high)
# DCT
def generate_different_dct(Images, r):
    Images_freq_low = []
    Images_freq_high = []
    low_mask = np.zeros((32,32))
    low_mask[0:r,0:r] = 1
    high_mask = 1-low_mask
    for i in tqdm(range(Images.shape[0])):
        img = np.float32(Images[i])
        tmp_low = np.zeros_like(img)
        tmp_high = np.zeros_like(img)
        for j in range(3):
            dct_im = cv2.dct(np.float32(img[:,:,j]))
            tmp_low[:,:,j] = dct_im*low_mask
            tmp_high[:,:,j] = dct_im*high_mask
        Images_freq_low.append(tmp_low)
        Images_freq_high.append(tmp_high)
    return np.array(Images_freq_low), np.array(Images_freq_high)



# cifar10
data = np.load('/data/CIFAR10/train_data.npy') # 50000*32*32*3
label = np.load('/data/CIFAR10/train_label.npy') # 50000
print(data.shape)
print(label.shape)
r = 12
low_data, high_data = generate_different_dft(data, r)
print(low_data.shape)
print(high_data.shape)

num_class = 10
label_dict = {i:[] for i in range(num_class)}
for i in range(len(label)):
    la = label[i]
    label_dict[la].append(low_data[i])
for i in range(num_class):
    print(len(label_dict[i]))


multi_data = []
class_arr = [i for i in range(10)]
for i in tqdm(range(len(label))):
    high_f = high_data[i]
    rand_la = np.random.choice(class_arr)
    while len(label_dict[rand_la])==0 or label[i]==rand_la:
        rand_la = np.random.choice(class_arr)
#     for c in range(10):
#         if len(label_dict[c])!=0 and label[i]!=c:
#             rand_la = c
#             break
    rand_index = random.randint(0,len(label_dict[rand_la])-1)
    low_f =  label_dict[rand_la][rand_index]
    tmp = np.zeros([32, 32, 3])
    for i in range(3):
        dct_im_1 = low_f[:,:,i]
        dct_im_2 = high_f[:,:,i]

        dct_im_1 += dct_im_2
        idct_im  = np.real(ifftshift(dct_im_1))
        idct_im = normal_img(idct_im)
        tmp[:,:,i] = idct_im
    tmp = np.uint8(tmp*255)
    del label_dict[rand_la][rand_index]
    if len(label_dict[rand_la])==0:
        class_arr.remove(rand_la)
    multi_data.append(tmp)
multi_data = np.array(multi_data)
print(multi_data.shape)
print('ok')
np.save("hybrid_high_12.npy", multi_data)
np.save("train_label.npy", label)
