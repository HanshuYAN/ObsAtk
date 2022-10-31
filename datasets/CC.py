#-**********************************************************************************-#
# Paired data are available
#   unpaired noisy and clean images, with different filenames
#-**********************************************************************************-#
import random, os, sys
import numpy as np
from numpy.lib.shape_base import vsplit
import scipy.misc as misc
import scipy.io as sio

import json

import skimage
import skimage.io as io

import torch
import torch.utils.data as data

sys.path.append("../")

import datasets.misc as misc
from utils import check_file_ext, makedirs


class CCreal(data.Dataset):
    def __init__(self, root='./data/RGB/MCWNNM-ICCV2017/Real_ccnoise_denoised_part', ext='.png',
                ndim=3, patch_size=None, isAug=False):
        super().__init__()
        self.root = root
        self.samples = self._scan_files(root, ext)
        
        self.ndim = ndim
        self.patch_size = patch_size
        self.isAug = isAug

    
    def _scan_files(self, filedir, ext):
        samples = []
        samples = sorted(
            [os.path.join(filedir, x) for x in os.listdir(filedir) if check_file_ext(x, ext)])
        noisy_samles = sorted([x for x in samples if x.endswith('_real.png')])
        gt_samles = sorted([x for x in samples if x.endswith('_mean.png')])
        return [(n,c) for n,c in zip(noisy_samles, gt_samles)]
    
    def __len__(self):
        return len(self.samples)

    
    def __getitem__(self, idx):
        """
        Args:
            idx (int): idx
        Returns:
            tuple: (noisy, clean)
        """
        noisy_file, clean_file = self.samples[idx]
            
        input_img = skimage.io.imread(noisy_file); assert input_img.dtype == 'uint8'
        input_img = skimage.img_as_float32(input_img)
        target_img = skimage.io.imread(clean_file); assert target_img.dtype == 'uint8'
        target_img = skimage.img_as_float32(target_img)
        
        
        input_img, target_img = misc.set_ndim_np([input_img, target_img], self.ndim)
        if self.patch_size is not None:
            input_img, target_img = misc.get_patch_np([input_img, target_img], self.patch_size, isPair=True)
        if self.isAug:
            input_img, target_img = misc.augment_np([input_img, target_img])
        input_img, target_img = misc.bhwc2bchw_np([input_img, target_img])
        return torch.from_numpy(input_img), noisy_file, torch.from_numpy(target_img), clean_file
    

