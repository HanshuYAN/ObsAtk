#-**********************************************************************************-#
# Paired data are available
#   unpaired noisy and clean images, with different filenames
#-**********************************************************************************-#
import random, os, sys
import numpy as np
from numpy.lib.shape_base import vsplit
import scipy.misc as misc

import skimage
import skimage.io as io

import torch
import torch.utils.data as data

sys.path.append("../")
import datasets.misc as misc
from utils import check_file_ext, makedirs


class PairedFolders(data.Dataset):
    """ Base dataset for RGB images. """
    def __init__(self, dir_clean, dir_noisy, ext='.png', is_bin=True, ndim=3, patch_size=None, isAug=False, repeat=1):
        super(PairedFolders, self).__init__()
        print('==> Dataset: dir_C, dir_N')
        print(dir_clean)
        print(dir_noisy)
        self.ext = ext
        self.is_bin = is_bin
        self.ndim = ndim
        self.patch_size = patch_size
        self.isAug = isAug
        self.repeat = repeat
        self.input_paths, self.target_paths = self._scan(dir_noisy, dir_clean, ext)
        if is_bin:
            dir_clean_npy = os.path.join(dir_clean,'npy')
            dir_noisy_npy = os.path.join(dir_noisy,'npy')
            makedirs(dir_clean_npy)
            makedirs(dir_noisy_npy)
            input_paths_bin, target_paths_bin = self._scan(dir_noisy_npy, dir_clean_npy, '.npy')
            if len(input_paths_bin) < len(self.input_paths):
                print('Preparing input binary files...')
                for v in self.input_paths:
                    print(v)
                    img = skimage.io.imread(v); assert img.dtype == 'uint8'  # skimage.io.imread img, default 8bit uint
                    v_split = v.split('/')
                    filename = v_split[-1]
                    path = '/'.join(v_split[:-1])
                    filename = filename.replace(ext, '.npy')
                    np.save(os.path.join(path, 'npy', filename), img)
            if len(target_paths_bin) < len(self.target_paths):
                print('Preparing target binary files...')
                for v in self.target_paths:
                    print(v)
                    img = skimage.io.imread(v); assert img.dtype == 'uint8'
                    v_split = v.split('/')
                    filename = v_split[-1]
                    path = '/'.join(v_split[:-1])
                    filename = filename.replace(ext, '.npy')
                    np.save(os.path.join(path, 'npy', filename), img)
            self.input_paths, self.target_paths = self._scan(os.path.join(dir_noisy,'npy'), os.path.join(dir_clean,'npy'), '.npy')
            print('Seperated binary files loaded!')
        else:
            print('Image ext files loaded!')

    def _scan(self, dir_input, dir_target, ext):
        input_paths = []; target_paths = []
        input_paths = sorted(
            [os.path.join(dir_input, x) for x in os.listdir(dir_input) if check_file_ext(x, ext)])
        target_paths = sorted(
            [os.path.join(dir_target, x) for x in os.listdir(dir_target) if check_file_ext(x, ext)])
        if ext != '.npy':
            assert len(input_paths) == len(target_paths)
        return input_paths, target_paths

    def __len__(self):
        return len(self.input_paths) * self.repeat

    def _get_index(self, idx):
        return idx % len(self.input_paths)
    
    def _load_file(self, idx):
        idx = self._get_index(idx)
        i_path = self.input_paths[idx]
        t_path = self.target_paths[idx]
        assert os.path.split(i_path)[-1] == os.path.split(t_path)[-1]            
        if self.is_bin:
            i_img = np.load(i_path); assert i_img.dtype == 'uint8'
            t_img = np.load(t_path); assert t_img.dtype == 'uint8'
        else:
            i_img = skimage.io.imread(i_path); assert i_img.dtype == 'uint8'
            t_img = skimage.io.imread(t_path); assert t_img.dtype == 'uint8'
        i_img = skimage.img_as_float32(i_img) # make sure output image is float32
        t_img = skimage.img_as_float32(t_img)
        return i_img, t_img, i_path, t_path
    
    def __getitem__(self, idx):
        input_img, target_img, input_path, target_path = self._load_file(idx)
        input_img, target_img = misc.set_ndim_np([input_img, target_img], self.ndim)
        if self.patch_size is not None:
            input_img, target_img = misc.get_patch_np([input_img, target_img], self.patch_size, isPair=True)
        if self.isAug:
            input_img, target_img = misc.augment_np([input_img, target_img])
        input_img, target_img = misc.bhwc2bchw_np([input_img, target_img])
        return torch.from_numpy(input_img), input_path, torch.from_numpy(target_img), target_path
    
    
class SingleFolder(data.Dataset):
    """ Base dataset for RGB images. """
    def __init__(self, dir_clean, ext='.png', is_bin=True, ndim=3, patch_size=40, isAug=False, isScaling=False, repeat=1):
        super(SingleFolder, self).__init__()
        print('==> Dataset: dir_C, dir_N')
        print(dir_clean)
        self.ext = ext
        self.is_bin = is_bin
        self.ndim = ndim
        self.patch_size = patch_size
        self.isAug = isAug
        self.isScaling = isScaling
        self.repeat = repeat
        self.paths = self._scan(dir_clean, ext)
        if is_bin:
            dir_clean_npy = os.path.join(dir_clean,'npy')
            makedirs(dir_clean_npy)
            paths_bin = self._scan(dir_clean_npy, '.npy')
            if len(paths_bin) < len(self.paths):
                print('Preparing target binary files...')
                for v in self.paths:
                    print(v)
                    img = skimage.io.imread(v); assert img.dtype == 'uint8'
                    v_split = v.split('/')
                    filename = v_split[-1]
                    path = '/'.join(v_split[:-1])
                    filename = filename.replace(ext, '.npy')
                    np.save(os.path.join(path, 'npy', filename), img)
            self.paths = self._scan(os.path.join(dir_clean,'npy'), '.npy')
            print('Seperated binary files loaded!')
        else:
            print('Image ext files loaded!')

    def _scan(self, dir_target, ext):
        paths = []
        paths = sorted(
            [os.path.join(dir_target, x) for x in os.listdir(dir_target) if check_file_ext(x, ext)])
        return paths

    def __len__(self):
        return len(self.paths) * self.repeat

    def _get_index(self, idx):
        return idx % len(self.paths)
    
    def _load_file(self, idx):
        idx = self._get_index(idx)
        path = self.paths[idx]
        if self.is_bin:
            img = np.load(path); assert img.dtype == 'uint8'
        else:
            img = skimage.io.imread(path); assert img.dtype == 'uint8'
        img = skimage.img_as_float32(img); assert img.dtype == 'float32'
        return img, path
    
    def __getitem__(self, idx):
        img, path = self._load_file(idx)
        if self.isScaling:
            img = misc.random_scaling([img])[0]
        img = misc.set_ndim_np([img], self.ndim)[0]
        if self.patch_size is not None:
            img = misc.get_patch_np([img], self.patch_size, isPair=True)[0]
        if self.isAug:
            img = misc.augment_np([img])[0]
        img = misc.bhwc2bchw_np([img])[0]
        return torch.from_numpy(img), path
    




    

from .SIDD import *
from .Fluorescence import *
from .PolyU import *
from .CC import *