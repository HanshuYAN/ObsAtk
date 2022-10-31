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




class FluorescenceGT(data.Dataset):
    """Class for the denoising dataset for both train and test, with 
    file structure:
        data_root/type/noise_level/fov/capture.png
        type:           12
        noise_level:    5 (+ 1: ground truth)
        fov:          20 (the 19th fov is for testing)
        capture.png:    50 images in each fov --> use fewer samples

    Args:
        root (str): root directory to the dataset
        train (bool): Training set if True, else Test set
        noise_levels (seq): e.g. [1, 2, 4] select `raw`, `avg2`, `avg4` folders
        types (seq, optional): e.g. ['TwoPhoton_BPAE_B', 'Confocal_MICE`]
        test_fov (int, optional): default 19. 19th fov is test fov
        captures (int): select # images within one folder
        transform (callable, optional): A function/transform that takes in 
            an PIL image and returns a transformed version. E.g, transforms.RandomCrop
        target_transform (callable, optional): A function/transform that takes 
            in the target and transforms it.
        loader (callable, optional): image loader
    """
    def __init__(self, root='./data/denoising-fluorescence/denoising/dataset', 
                 train=False, types=None, test_fov=[19], 
                 repeat=1, ndim=3, patch_size=None, isAug=False, isMemory=True):
        super().__init__()
        all_types = ['TwoPhoton_BPAE_R', 'TwoPhoton_BPAE_G', 'TwoPhoton_BPAE_B',
            'TwoPhoton_MICE', 'Confocal_MICE', 'Confocal_BPAE_R',
            'Confocal_BPAE_G', 'Confocal_BPAE_B', 'Confocal_FISH',
            'WideField_BPAE_R', 'WideField_BPAE_G', 'WideField_BPAE_B']
        if types is None:
            self.types = all_types
        else:
            assert all([img_type in all_types for img_type in types])
            self.types = types
        self.root = root
        self.train = train
        if train:
            fovs = list(range(1, 20+1))
            for tf in test_fov:
                fovs.remove(tf)
            self.fovs = fovs
        else:
            self.fovs = test_fov
        self.samples = self._scan_files()
        
        self.repeat = repeat
        self.ndim = ndim
        self.patch_size = patch_size
        self.isAug = isAug
        
        self.isMemory = isMemory
        if isMemory:
            self._load_into_memory()
        

    def _scan_files(self, ext='.png'):
        samples = []
        root_dir = os.path.expanduser(self.root)
        subdirs = [os.path.join(root_dir, name) for name in os.listdir(root_dir)
            if (os.path.isdir(os.path.join(root_dir, name)) and name in self.types)]
        subdirs = sorted(subdirs)

        for subdir in subdirs:
            gt_dir = os.path.join(subdir, 'gt')
            for i_fov in self.fovs:
                clean_file = os.path.join(gt_dir, f'{i_fov}', 'avg50.png')
                samples.append(clean_file)
        return samples
    
    def _load_into_memory(self):
        self.sample_files = []
        for fname in self.samples:
            target_img = skimage.io.imread(fname)
            assert target_img.dtype == 'uint8'
            self.sample_files.append(target_img)
    
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
        if self.isMemory:
            clean_file = self.samples[idx]
            target_img = self.sample_files[idx]
        else:
            clean_file = self.samples[idx]
            target_img = skimage.io.imread(clean_file)
            
        assert target_img.dtype == 'uint8'
        target_img = skimage.img_as_float32(target_img)
        
        target_img, *_ = misc.set_ndim_np([target_img], self.ndim)
        if self.patch_size is not None:
            target_img, *_ = misc.get_patch_np([target_img], self.patch_size, isPair=True)
        if self.isAug:
            target_img, *_ = misc.augment_np([target_img])
        target_img, *_ = misc.bhwc2bchw_np([target_img])
        return torch.from_numpy(target_img), clean_file


class FluorescenceTestMix(data.Dataset):
    """Class for the denoising dataset for both train and test, with 
    file structure:
        data_root/type/noise_level/fov/capture.png
        type:           12
        noise_level:    5 (+ 1: ground truth)
        fov:          20 (the 19th fov is for testing)
        capture.png:    50 images in each fov --> use fewer samples

    Args:
        root (str): root directory to the dataset
        train (bool): Training set if True, else Test set
        noise_levels (seq): e.g. [1, 2, 4] select `raw`, `avg2`, `avg4` folders
        types (seq, optional): e.g. ['TwoPhoton_BPAE_B', 'Confocal_MICE`]
        test_fov (int, optional): default 19. 19th fov is test fov
        captures (int): select # images within one folder
        transform (callable, optional): A function/transform that takes in 
            an PIL image and returns a transformed version. E.g, transforms.RandomCrop
        target_transform (callable, optional): A function/transform that takes 
            in the target and transforms it.
        loader (callable, optional): image loader
    """
    def __init__(self, root='./data/denoising-fluorescence/denoising/dataset', noise_levels=[1],
                ndim=3, patch_size=None, isAug=False):
        super().__init__()
        all_noise_levels = [1, 2, 4, 8, 16]
        all_types = ['test_mix']
        assert all([level in all_noise_levels for level in noise_levels])
        self.noise_levels = noise_levels
        self.root = root
        self.samples = self._scan_files()
        
        self.ndim = ndim
        self.patch_size = patch_size
        self.isAug = isAug

    def _scan_files(self):
        samples = []
        root_dir = os.path.expanduser(self.root)
        subdirs = [os.path.join(root_dir, 'test_mix')]
        subdirs = sorted(subdirs)
        
        for subdir in subdirs:
            gt_dir = os.path.join(subdir, 'gt')
            for noise_level in self.noise_levels:
                if noise_level == 1:
                    noise_dir = os.path.join(subdir, 'raw')
                elif noise_level in [2, 4, 8, 16]:
                    noise_dir = os.path.join(subdir, f'avg{noise_level}')

                for fname in sorted(os.listdir(noise_dir)):
                    if check_file_ext(fname, '.png'):
                        noisy_file = os.path.join(noise_dir, fname)
                        clean_file = os.path.join(gt_dir, fname)
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


    
  
    
    
    

class FluorescenceFolder(data.Dataset):
    """Class for the denoising dataset for both train and test, with 
    file structure:
        data_root/type/noise_level/fov/capture.png
        type:           12
        noise_level:    5 (+ 1: ground truth)
        fov:          20 (the 19th fov is for testing)
        capture.png:    50 images in each fov --> use fewer samples

    Args:
        root (str): root directory to the dataset
        train (bool): Training set if True, else Test set
        noise_levels (seq): e.g. [1, 2, 4] select `raw`, `avg2`, `avg4` folders
        types (seq, optional): e.g. ['TwoPhoton_BPAE_B', 'Confocal_MICE`]
        test_fov (int, optional): default 19. 19th fov is test fov
        captures (int): select # images within one folder
        transform (callable, optional): A function/transform that takes in 
            an PIL image and returns a transformed version. E.g, transforms.RandomCrop
        target_transform (callable, optional): A function/transform that takes 
            in the target and transforms it.
        loader (callable, optional): image loader
    """
    def __init__(self, root='./data/denoising-fluorescence/denoising/dataset', 
                 train=False, noise_levels=[1], types=['TwoPhoton_MICE'], test_fov=[19], captures=50, 
                 repeat=1, ndim=3, patch_size=None, isAug=False):
        super().__init__()
        all_noise_levels = [1, 2, 4, 8, 16]
        all_types = ['TwoPhoton_BPAE_R', 'TwoPhoton_BPAE_G', 'TwoPhoton_BPAE_B',
            'TwoPhoton_MICE', 'Confocal_MICE', 'Confocal_BPAE_R',
            'Confocal_BPAE_G', 'Confocal_BPAE_B', 'Confocal_FISH',
            'WideField_BPAE_R', 'WideField_BPAE_G', 'WideField_BPAE_B']
        assert all([level in all_noise_levels for level in noise_levels])
        self.noise_levels = noise_levels
        if types is None:
            self.types = all_types
        else:
            assert all([img_type in all_types for img_type in types])
            self.types = types
        self.root = root
        self.train = train
        if train:
            fovs = list(range(1, 20+1))
            for tf in test_fov:
                fovs.remove(tf)
            self.fovs = fovs
        else:
            self.fovs = test_fov
        self.captures = captures
        self.samples = self._scan_files()
        
        self.repeat = repeat
        self.ndim = ndim
        self.patch_size = patch_size
        self.isAug = isAug

    def _scan_files(self):
        samples = []
        root_dir = os.path.expanduser(self.root)
        subdirs = [os.path.join(root_dir, name) for name in os.listdir(root_dir)
            if (os.path.isdir(os.path.join(root_dir, name)) and name in self.types)]
        subdirs = sorted(subdirs)
        
        for subdir in subdirs:
            gt_dir = os.path.join(subdir, 'gt')
            for noise_level in self.noise_levels:
                if noise_level == 1:
                    noise_dir = os.path.join(subdir, 'raw')
                else:
                    noise_dir = os.path.join(subdir, f'avg{noise_level}')
                    
                for i_fov in self.fovs:
                    noisy_fov_dir = os.path.join(noise_dir, f'{i_fov}')
                    clean_file = os.path.join(gt_dir, f'{i_fov}', 'avg50.png')
                    if self.train:
                        noisy_captures = []
                        for fname in sorted(os.listdir(noisy_fov_dir))[:self.captures]:
                            if check_file_ext(fname, '.png'):
                                noisy_file = os.path.join(noisy_fov_dir, fname)
                                noisy_captures.append(noisy_file)
                        # randomly select one noisy capture when loading from FOV     
                        samples.append((noisy_captures, clean_file))
                    else:
                        # for test, only one FOV, use all of them
                        for fname in sorted(os.listdir(noisy_fov_dir))[:self.captures]:
                            if check_file_ext(fname, '.png'):
                                noisy_file = os.path.join(noisy_fov_dir, fname)
                                samples.append((noisy_file, clean_file))
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
        if self.train:
            noisy_captures, clean_file = self.samples[idx]
            idx = np.random.choice(len(noisy_captures), 1)
            noisy_file = noisy_captures[idx[0]]
        else:
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