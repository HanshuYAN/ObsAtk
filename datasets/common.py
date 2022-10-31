import random

import numpy as np
import skimage.io as sio
import skimage.color as sc
# import skimage.transform as st

# import scipy.ndimage.filters as filters

import torch
# from torchvision import transforms
"""
def get_patch(img_in, img_tar, patch_size):
    h ,w= img_in.shape[:2]
    x = random.randint(0, w - patch_size)
    y = random.randint(0, h - patch_size)
    img_in = img_in[y:y+patch_size, x:x+patch_size, :]

    h ,w = img_tar.shape[:2]
    x = random.randint(0, w - patch_size)
    y = random.randint(0, h - patch_size)
    img_tar = img_tar[y:y+patch_size, x:x+patch_size, :]

    return img_in, img_tar

"""
def get_patch(img_in, img_tar, patch_size, isPair):
    if isPair:
        assert img_in.shape == img_tar.shape
        h ,w= img_in.shape[:2]
        x = random.randint(0, w - patch_size)
        y = random.randint(0, h - patch_size)
        img_in = img_in[y:y+patch_size, x:x+patch_size]
        img_tar = img_tar[y:y+patch_size, x:x+patch_size]
    else:    
        h ,w= img_in.shape[:2]
        x = random.randint(0, w - patch_size)
        y = random.randint(0, h - patch_size)
        img_in = img_in[y:y+patch_size, x:x+patch_size]

        h ,w = img_tar.shape[:2]
        x = random.randint(0, w - patch_size)
        y = random.randint(0, h - patch_size)
        img_tar = img_tar[y:y+patch_size, x:x+patch_size]

    return img_in, img_tar

# def dtype_mean_shift(img, rgb_range, rgb_mean, rgb_std, mode):
#     # img.shape is (h,w,channel), 0-255
#     assert (mode == '+') or (mode == '-')
#     assert (img.shape[2] == len(rgb_mean)) and (len(rgb_mean) == len(rgb_std))

#     rgb_mean = np.resize(np.array(rgb_mean),(1,1,3))
#     rgb_std = np.resize(np.array(rgb_std),(1,1,3))    
#     if mode == '-':
#         # 8bit to float, de-mean, div_std
#         img = img / 255.0 * rgb_range
#         img = np.divide(np.subtract(img,rgb_mean),rgb_std)
#     else:
#         # mul_std, add_mean, float to 8bit
#         img = np.add(np.multiply(img, rgb_std), rgb_mean)
#         img = img / rgb_range * 255.0
#         img = np.clip(img, 0, 255)
#         img = img.astype(np.uint8)
#     return img

def dtype_mean_shift_gray(img, rgb_range, rgb_mean, rgb_std, mode):
    # img.shape is (h,w,1), 0-255
    assert (mode == '+') or (mode == '-')    
    assert (img.shape[2] == len(rgb_mean)) and (len(rgb_mean) == len(rgb_std))
    
    rgb_mean = np.array(rgb_mean)
    rgb_std = np.array(rgb_std) 
    if mode == '-':
        # 8bit to float, de-mean, div_std
        img = img / 255.0 * rgb_range
        img = np.divide(np.subtract(img,rgb_mean),rgb_std)
    else:
        # mul_std, add_mean, float to 8bit
        # assert img.max()<=1
        img = np.add(np.multiply(img, rgb_std), rgb_mean)
        img = img / rgb_range * 255.0
        img = np.clip(img, 0, 255)
        img = img.astype(np.uint8)
    return img

def set_channel(l, n_channel):
    def _set_channel(img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        c = img.shape[2]
        if n_channel == 1 and c == 3:
            # need gray, so do rgb to gray
            img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
        elif n_channel == 3 and c == 1:
            # need rgb, so replicate gray to 3 channels
            img = np.concatenate([img] * n_channel, 2)

        return img

    return [_set_channel(_l) for _l in l]

def np2Tensor(l):
    # img.shape is (h,w,channel)
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))        
        tensor = torch.from_numpy(np_transpose).float()
        # tensor.mul_(rgb_range / 255)
        return tensor
    return [_np2Tensor(_l) for _l in l]

def add_noise(x, noise='.'):
    if noise is not '.':
        noise_type = noise[0]
        noise_value = int(noise[1])
        if noise_type == 'G':
            noises = np.random.normal(scale=noise_value, size=x.shape)
            noises = noises.round()
        elif noise_type == 'S':
            noises = np.random.poisson(x * noise_value) / noise_value
            noises = noises - noises.mean(axis=0).mean(axis=0)

        x_noise = x.astype(np.int16) + noises.astype(np.int16)
        x_noise = x_noise.clip(0, 255).astype(np.uint8)
        return x_noise
    else:
        return x

def gaussian_kernel(kernel_size=5, sigma=5):
    t = np.linspace(-5,5,kernel_size)
    bump = np.exp(-1/(sigma**2)*(t**2))
    bump /= np.trapz(bump)
    kernel = bump[:,np.newaxis] * bump[np.newaxis,:]
    return kernel

def augment(l, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        # torch.from_numpy do not support negative strides
        if hflip: img = img[:, ::-1, :].copy()
        if vflip: img = img[::-1, :, :].copy()
        if rot90: img = img.transpose(1, 0, 2).copy()
        return img

    return [_augment(_l) for _l in l]





import os
import sys
import pathlib
import numpy as np

import torch


path_of_this_module = os.path.dirname(sys.modules[__name__].__file__) # the dir including this file

def check_file_ext(x, ext='.png', ignore_prefix='._'):
    return True if x.endswith(ext) and not(x.startswith(ignore_prefix)) else False

#####################
## Numpy operations
#####################

def bchw2bhwc(x):
    if isinstance(x, np.ndarray):
        pass
    else:
        raise

    if x.ndim == 3:
        return np.moveaxis(x, 0, 2)
    if x.ndim == 4:
        return np.moveaxis(x, 1, 3)


def bhwc2bchw(x):
    if isinstance(x, np.ndarray):
        pass
    else:
        raise

    if x.ndim == 3:
        return np.moveaxis(x, 2, 0)
    if x.ndim == 4:
        return np.moveaxis(x, 3, 1)


def _imshow(img):
    import matplotlib.pyplot as plt
    img = bchw2bhwc(img.detach().cpu().numpy())
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    plt.imshow(img, vmin=0, vmax=1)
    plt.axis("off")


#
# Image Processing 
#
import random, os

import numpy as np
import skimage.io as sio
import skimage.color as sc


def is_img(x, ext='.jpg'):
    if x.endswith(ext) and not(x.startswith('._')):
        return True
    else:
        return False

def dir_scan(img_dir, ext):
    img_list = sorted(
        [os.path.join(img_dir, x) 
                for x in os.listdir(img_dir) if is_img(x, ext)]
        )
    return img_list

def add_noise_to_img(x, noise=None):
    # x, float 0-1
    # noise, None or ['G', 15]
    assert len(x.shape)==4
    if noise is not None:
        noise_type = noise[0]
        noise_value = int(noise[1])
        noise = torch.randn(x.shape) * noise_value / 255
        x_noise = x + noise
        return torch.clamp(x_noise, min=0, max=1)
    else:
        return x

def add_noise_to_img_np(x, noise=['G', 15/255]):
    noise_type = noise[0]
    noise_value = noise[1]
    if noise_type == 'G':
        noises = np.random.normal(scale=noise_value, size=x.shape)
    else:
        noises = np.random.poisson(x * noise_value) / noise_value
        noises = noises - noises.mean(axis=0).mean(axis=0)
    x_noise = x + noises
    x_noise = x_noise.clip(0, 1)
    return x_noise

    
    
def get_patch(img_list, patch_size, isPair):
    assert len(img_list) <= 2, "at most 2 images"
    
    if len(img_list) == 1:
        img = img_list[0]
        h ,w= img.shape[:2]
        randw = random.randint(0, w - patch_size)
        randh = random.randint(0, h - patch_size)
        img = img[randh:randh+patch_size, randw:randw+patch_size]
        return [img]
    else:
        img_in = img_list[0]
        img_tar = img_list[1]
        if isPair:
            assert img_in.shape == img_tar.shape
            h ,w= img_in.shape[:2]
            randw = random.randint(0, w - patch_size)
            randh = random.randint(0, h - patch_size)
            img_in = img_in[randh:randh+patch_size, randw:randw+patch_size]
            img_tar = img_tar[randh:randh+patch_size, randw:randw+patch_size]
        else:    
            h ,w= img_in.shape[:2]
            randw = random.randint(0, w - patch_size)
            randh = random.randint(0, h - patch_size)
            img_in = img_in[randh:randh+patch_size, randw:randw+patch_size]

            h ,w = img_tar.shape[:2]
            randw = random.randint(0, w - patch_size)
            randh = random.randint(0, h - patch_size)
            img_tar = img_tar[randh:randh+patch_size, randw:randw+patch_size]

        return [img_in, img_tar]


    
def dtype_mean_shift(img_list, rgb_range, rgb_mean, rgb_std, mode):
    # (h,w,colors)
    # -: uint8 to rgb_range, then normalize
    # +: reverse-normalization, rgb_range to uint8
    assert (mode == '+') or (mode == '-')
    
    def _dtype_mean_shift(img, rgb_range=rgb_range, rgb_mean=rgb_mean, rgb_std=rgb_std, mode=mode):
        n_colors = img.shape[2]
        assert (n_colors == len(rgb_mean)) and (len(rgb_mean) == len(rgb_std))
        
        if n_colors == 3:
            rgb_mean = np.resize(np.array(rgb_mean),(1,1,3))
            rgb_std = np.resize(np.array(rgb_std),(1,1,3))
        else:
            rgb_mean = np.array(rgb_mean)
            rgb_std = np.array(rgb_std)
            
        if mode == '-':
            # 8bit to float, de-mean, div_std
            img = img / 1 * rgb_range
            img = np.divide(np.subtract(img,rgb_mean),rgb_std)
        else:
            # mul_std, add_mean, float to 8bit
            img = np.add(np.multiply(img, rgb_std), rgb_mean)
            img = img / rgb_range * 1
            img = np.clip(img, 0, 1)
            # img = img.astype(np.uint8)
        return img
    
    return [_dtype_mean_shift(_l) for _l in img_list]


def set_channel(l, n_channel):
    def _set_channel(img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        c = img.shape[2]
        if n_channel == 1 and c == 3:
            # need gray, so do rgb to gray
            img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
        elif n_channel == 3 and c == 1:
            # need rgb, so replicate gray to 3 channels
            img = np.concatenate([img] * n_channel, 2)

        return img

    return [_set_channel(_l) for _l in l]


def np2Tensor(l):
    # (h,w,channel) to (channel, h, w)
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))        
        tensor = torch.from_numpy(np_transpose).float()
        return tensor
    return [_np2Tensor(_l) for _l in l]


def tensor2im(input_image, rgb_range, rgb_mean, rgb_std):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    # Tensor with shape (1, channel, h, w)
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = np.transpose(image_numpy, (1, 2, 0))  
    image_numpy = dtype_mean_shift([image_numpy], 
                                        rgb_range=rgb_range, 
                                        rgb_mean=rgb_mean, 
                                        rgb_std=rgb_std, 
                                        mode='+')
    image_numpy = image_numpy[0]
    if image_numpy.shape[2] == 1:
        h = image_numpy.shape[0]; w = image_numpy.shape[1]
        image_numpy = image_numpy.reshape(h,w)
    # output: (h,w) or (h,w,3)
    return image_numpy


# def gaussian_kernel(kernel_size=5, sigma=5):
#     t = np.linspace(-5,5,kernel_size)
#     bump = np.exp(-1/(sigma**2)*(t**2))
#     bump /= np.trapz(bump)
#     kernel = bump[:,np.newaxis] * bump[np.newaxis,:]
#     return kernel

def augment(l, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        # torch.from_numpy do not support negative strides
        if hflip: img = img[:, ::-1, :].copy()
        if vflip: img = img[::-1, :, :].copy()
        if rot90: img = img.transpose(1, 0, 2).copy()
        return img

    return [_augment(_l) for _l in l]
