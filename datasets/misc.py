import os, random
import numpy as np
from numpy.lib.arraysetops import isin
import skimage
from skimage.transform import rescale
import torch


def bchw2bhwc_np(l):
    def _bchw2bhwc(x):
        if isinstance(x, np.ndarray):
            pass
        else:
            raise
        if x.ndim == 3:
            return np.moveaxis(x, source=0, destination=2)
        if x.ndim == 4:
            return np.moveaxis(x, source=1, destination=3)
    return [_bchw2bhwc(_l) for _l in l]



def bhwc2bchw_np(l):
    def _bhwc2bchw(x):
        if isinstance(x, np.ndarray):
            pass
        else:
            raise

        if x.ndim == 3:
            return np.moveaxis(x, source=2, destination=0)
        if x.ndim == 4:
            return np.moveaxis(x, source=3, destination=1)
    return [_bhwc2bchw(_l) for _l in l]


def Tensor2Img(img_tensor):
    assert isinstance(img_tensor, torch.Tensor)
    img_np = img_tensor.detach().cpu().numpy()
    return np.transpose(img_np, (1,2,0))

#################################
## Image Processing
#################################

def random_scaling(l):
    scales = [1, 0.9, 0.8, 0.7]
    def _random_scaling(img):
        scale = random.choice(scales)
        # out = rescale(img, scale, anti_aliasing=False); assert out.dtype=='float32'
        out = rescale(img, scale, order=3, anti_aliasing=False); assert out.dtype=='float32'
        return out
    return [_random_scaling(_l) for _l in l]


def set_ndim_np(l, ndim):
    def _set_ndim(img):
        assert img.ndim in [2,3]
        if img.ndim == 2 and ndim == 3:
            img = np.expand_dims(img, axis=2)
        elif img.ndim == 3 and ndim == 2:
            img = img[:,:,0]
        return img
    return [_set_ndim(_l) for _l in l]

def get_patch_np(img_list, patch_size, isPair):
    assert img_list[0].ndim == 3
    out_img_list = []
    if isPair:
        img_shape = img_list[0].shape
        h ,w= img_shape[:2]
        randw = random.randint(0, w - patch_size)
        randh = random.randint(0, h - patch_size)
        for img in img_list:
            assert img.shape == img_shape
            out_img_list.append(img[randh:randh+patch_size, randw:randw+patch_size])
    else:
        for img in img_list:
            h ,w= img.shape[:2]
            randw = random.randint(0, w - patch_size)
            randh = random.randint(0, h - patch_size)
            out_img_list.append(img[randh:randh+patch_size, randw:randw+patch_size])
    return out_img_list


def augment_np(l):
    def _augment(img):
        mode = random.choice([0,1,2,3,4,5,6,7])
        out = img
        if mode == 0:
            # original
            out = out
        elif mode == 1:
            # flip up and down
            out = np.flipud(out)
        elif mode == 2:
            # rotate counterwise 90 degree
            out = np.rot90(out)
        elif mode == 3:
            # rotate 90 degree and flip up and down
            out = np.rot90(out)
            out = np.flipud(out)
        elif mode == 4:
            # rotate 180 degree
            out = np.rot90(out, k=2)
        elif mode == 5:
            # rotate 180 degree and flip
            out = np.rot90(out, k=2)
            out = np.flipud(out)
        elif mode == 6:
            # rotate 270 degree
            out = np.rot90(out, k=3)
        elif mode == 7:
            # rotate 270 degree and flip
            out = np.rot90(out, k=3)
            out = np.flipud(out)
        return out.copy()
    return [_augment(_l) for _l in l]


def normalize_np(img_list, rgb_mean, rgb_std, mode):
    assert (mode == '+') or (mode == '-')
    def _normalize(img, rgb_mean=rgb_mean, rgb_std=rgb_std, mode=mode):
        n_colors = img.shape[2]
        assert (n_colors == len(rgb_mean)) and (len(rgb_mean) == len(rgb_std))
        if n_colors == 3:
            rgb_mean = np.resize(np.array(rgb_mean),(1,1,3))
            rgb_std = np.resize(np.array(rgb_std),(1,1,3))
        else:
            rgb_mean = np.array(rgb_mean)
            rgb_std = np.array(rgb_std)
            
        if mode == '-':
            img = np.divide(np.subtract(img,rgb_mean),rgb_std)
        else:
            img = np.add(np.multiply(img, rgb_std), rgb_mean)
            img = np.clip(img, 0, 1)
        return img
    return [_normalize(_l) for _l in img_list]


def add_noise_to_img_np(x, noise=['G', 15/255]):
    assert x.dtype != 'uint8'
    noise_type = noise[0]
    noise_value = noise[1]
    if noise_type == 'G':
        noises = np.random.normal(loc=0, scale=noise_value, size=x.shape)
    else:
        noises = np.random.uniform(-noise_value, noise_value, size=x.shape)
    x_noise = x + noises
    x_noise = x_noise.clip(0, 1)
    return x_noise

# Torch tensor
class adding_guas_noise():
    def __init__(self, param) -> None:
        if not isinstance(param, torch.Tensor):
            param = torch.tensor([param])
        self.param = param
    def perturb(self, x, verbose=None):
        x = x.detach().clone()
        assert len(x.shape)==4
        if self.param.shape[0]>1:
            assert self.param.shape[0] == x.shape[0]
        noise = torch.randn(x.shape) * self.param.view(-1,1,1,1)
        x_noise = x + noise.to(x)
        return torch.clamp(x_noise, min=0, max=1)
    
class adding_uniform_noise():
    def __init__(self, param) -> None:
        if not isinstance(param, torch.Tensor):
            param = torch.tensor([param])
        self.param = param
    def perturb(self, x, verbose=None):
        x = x.detach().clone()
        assert len(x.shape)==4
        if self.param.shape[0]>1:
            assert self.param.shape[0] == x.shape[0]
        noise = (torch.rand(x.shape) * 2 - 1) * self.param.view(-1,1,1,1)
        x_noise = x + noise.to(x)
        return torch.clamp(x_noise, min=0, max=1)    
    
class adding_poisson_noise():
    def __init__(self, param) -> None:
        if not isinstance(param, torch.Tensor):
            param = torch.tensor([param])
        self.param = param
    def perturb(self, x, verbose=None):
        x = x.detach().clone()
        assert len(x.shape)==4
        if self.param.shape[0]>1:
            assert self.param.shape[0] == x.shape[0]
        noise = torch.poisson(torch.ones(x.shape) * self.param.view(-1,1,1,1))
        noise = (noise - noise.mean()) * 1./255
        x_noise = x + noise.to(x)
        return torch.clamp(x_noise, min=0, max=1)
    
    

