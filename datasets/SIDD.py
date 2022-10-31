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
        
        
class SIDDpatches(data.Dataset):
    def __init__(self, root='./data/RGB/SIDD/Patch512S', is_bin=False,
                ndim=3, patch_size=None, isAug=False, repeat=1, gt_only=False):
        super().__init__()
        self.root = root
        self.ndim = ndim
        self.patch_size = patch_size
        self.isAug = isAug
        self.repeat = repeat
        self.gt_only = gt_only
        self.is_bin = is_bin
        
        self.samples = self._scan_files('.png')
        if is_bin:
            samples = self._scan_files(ext='.npy')
            if len(self.samples) == len(samples):
                self.samples = samples
                print('binary files alreay existed. using binary...')
            else:
                for (nfile, cfile) in self.samples:
                    img = skimage.io.imread(nfile); assert img.dtype == 'uint8'  # skimage.io.imread img, default 8bit uint
                    np.save(nfile.replace('.png', '.npy'), img)
                    img = skimage.io.imread(cfile); assert img.dtype == 'uint8'  # skimage.io.imread img, default 8bit uint
                    np.save(cfile.replace('.png', '.npy'), img)
                samples = self._scan_files(ext='.npy')
                assert len(self.samples) == len(samples)
                self.samples = samples
                print('binary files processed. using binary...')

    def _scan_files(self, ext='.png'):
        samples = []
        root = os.path.expanduser(self.root)
        subdirs = [os.path.join(root, name) for name in os.listdir(root)
            if (os.path.isdir(os.path.join(root, name)) and (not name.startswith('.')))]
        subdirs = sorted(subdirs)
        for subdir in subdirs:
            files = sorted(
                [os.path.join(subdir, x) for x in os.listdir(subdir) if check_file_ext(x, ext)])
            noisy_files = sorted([x for x in files if x.endswith('_noisy'+ext)])
            gt_files = sorted([x for x in files if x.endswith('_gt'+ext)])
            samples += list(zip(noisy_files, gt_files))
        return samples
    
    def __len__(self):
        return len(self.samples) * self.repeat

    def _get_index(self, idx):
        return idx % len(self.samples)
    
    def __getitem__(self, idx):
        """
        Args:
            idx (int): idx
        Returns:
            tuple: (noisy, clean)
        """
        idx = self._get_index(idx)
        noisy_file, clean_file = self.samples[idx]
        if self.is_bin:
            input_img = np.load(noisy_file); assert input_img.dtype == 'uint8'
            target_img = np.load(clean_file); assert target_img.dtype == 'uint8'
        else:
            input_img = skimage.io.imread(noisy_file); assert input_img.dtype == 'uint8'
            target_img = skimage.io.imread(clean_file); assert target_img.dtype == 'uint8'
        input_img = skimage.img_as_float32(input_img)
        target_img = skimage.img_as_float32(target_img)
        
        
        input_img, target_img = misc.set_ndim_np([input_img, target_img], self.ndim)
        if self.patch_size is not None:
            input_img, target_img = misc.get_patch_np([input_img, target_img], self.patch_size, isPair=True)
        if self.isAug:
            input_img, target_img = misc.augment_np([input_img, target_img])
        input_img, target_img = misc.bhwc2bchw_np([input_img, target_img])
    
        if self.gt_only:
            return torch.from_numpy(target_img), clean_file
        else:
            return torch.from_numpy(input_img), noisy_file, torch.from_numpy(target_img), clean_file
    
    
    
    
class SIDDsRGBVal(data.Dataset):
    def __init__(self, root='./data/RGB/SIDD/Validation', 
                ndim=3, patch_size=None, isAug=False, gt_only=False):
        super().__init__()
        # all_noise_levels = ['s25', 's20', 's15','s10','orig']
        # # assert all([level in all_noise_levels for level in noise_levels])
        # assert noise_levels in all_noise_levels
        # self.noise_levels = noise_levels
        self.root = root
        # self.samples = self._scan_files()
        
        self.noisy_imgs = sio.loadmat(os.path.join(root, 'ValidationNoisyBlocksSrgb.mat'))['ValidationNoisyBlocksSrgb']
        self.clean_imgs = sio.loadmat(os.path.join(root, 'ValidationGtBlocksSrgb.mat'))['ValidationGtBlocksSrgb']
        n_s, n_p, h, w, c = self.clean_imgs.shape
        # n_patches = n_s * n_p
        self.noisy_imgs = self.noisy_imgs.reshape(-1, h, w, c)
        self.clean_imgs = self.clean_imgs.reshape(-1, h, w, c)
        assert self.noisy_imgs.shape[0] == self.clean_imgs.shape[0]
        
        self.ndim = ndim
        self.patch_size = patch_size
        self.isAug = isAug
        
        self.gt_only = gt_only

    
    def __len__(self):
        # return len(self.samples)
        return self.clean_imgs.shape[0]

    
    def __getitem__(self, idx):
        """
        Args:
            idx (int): idx
        Returns:
            tuple: (noisy, clean)
        """
        # noisy_file, clean_file = self.samples[idx]
            
        # input_img = skimage.io.imread(noisy_file); 
        input_img = self.noisy_imgs[idx]
        assert input_img.dtype == 'uint8'
        input_img = skimage.img_as_float32(input_img)
        # target_img = skimage.io.imread(clean_file); 
        target_img = self.clean_imgs[idx]
        assert target_img.dtype == 'uint8'
        target_img = skimage.img_as_float32(target_img)
        
        
        input_img, target_img = misc.set_ndim_np([input_img, target_img], self.ndim)
        if self.patch_size is not None:
            input_img, target_img = misc.get_patch_np([input_img, target_img], self.patch_size, isPair=True)
        if self.isAug:
            input_img, target_img = misc.augment_np([input_img, target_img])
        input_img, target_img = misc.bhwc2bchw_np([input_img, target_img])
        
        if self.gt_only:
            return torch.from_numpy(target_img), idx
        else:
            return torch.from_numpy(input_img), idx, torch.from_numpy(target_img), idx
        
        
        
        
        
        

class SIDDsRGB(data.Dataset):
    def __init__(self, root='./data/RGB/SIDD/CropC512', noise_levels='orig',
                ndim=3, patch_size=None, isAug=False):
        super().__init__()
        all_noise_levels = ['s25', 's20', 's15','s10','orig']
        # assert all([level in all_noise_levels for level in noise_levels])
        assert noise_levels in all_noise_levels
        self.noise_levels = noise_levels
        self.root = root
        self.samples = self._scan_files()
        
        self.ndim = ndim
        self.patch_size = patch_size
        self.isAug = isAug

    def _scan_files(self):
        samples = []
        root_dir = os.path.expanduser(self.root)
        subdirs = [os.path.join(root_dir, name) for name in os.listdir(root_dir)
            if (os.path.isdir(os.path.join(root_dir, name)) and (not name.startswith('.')))]
        subdirs = sorted(subdirs)
        
        for subdir in subdirs:
            clean_file = os.path.join(subdir, 'ground_truth.png')
            noisy_file = os.path.join(subdir, f'noisy_{self.noise_levels}.png')
            samples.append((noisy_file, clean_file))          
        return samples
    
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