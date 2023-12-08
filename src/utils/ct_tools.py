import os
import torch
import numpy as np
import torchvision

class CT_Preprocessing:
    def __init__(self, miuwater=0.192):
        self.miuwater = miuwater

    def HU2miu(self, HUimg):
        miuimg = HUimg / 1000 * self.miuwater + self.miuwater
        return miuimg

    def miu2HU(self, miuimg):
        HUimg = (miuimg - self.miuwater) / self.miuwater * 1000
        return HUimg

    def CTrange(self, img, HUmode=True, minran=-1000, maxran=2000):
        assert minran < maxran
        if HUmode is False: 
            img = self.miu2HU(img)
        img[img<minran] = minran
        img[img>maxran] = maxran
        if HUmode is False:
            img = self.HU2miu(img)
        return img
    
    def window_transform(self, HUimg, width=3000, center=500, norm=False):
        minwindow = float(center) - 0.5 * float(width)
        winimg = (HUimg - minwindow) / float(width)
        winimg[winimg < 0] = 0
        winimg[winimg > 1] = 1
        if norm:
            print('normalize to 0-255')
            winimg = (winimg * 255).astype('float')
        return winimg

    def back_window_transform(self, winimg, width=3000, center=500, norm=False):
        ''' 01normalization -> HUimg
        '''
        minwindow = float(center) - 0.5 * float(width)
        if norm:
            winimg = winimg/255
        HUimg = winimg * float(width) + minwindow
        return HUimg


def save_ct(ct_image, path, **kwargs):
    ''''''
    torchvision.utils.save_image(ct_image, path, **kwargs)






# ==== add noise ====
# ==== add noise ====



def add_poisson_to_sinogram_torch(sinogram, IO, seed=None):
    # seed, i wont set seed
    max_sinogram = sinogram.max()
    sinogramRawScaled = sinogram / max_sinogram.max()
    # to detector count?
    sinogramCT = IO * torch.exp(-sinogramRawScaled)
    # add poison noise
    sinogram_CT_C = torch.zeros_like(sinogramCT)
    for i in range(sinogram_CT_C.shape[0]):
        for j in range(sinogram_CT_C.shape[1]):
            # sinogram_CT_C[i, j] = np.random.poisson(sinogramCT[i, j])
            sinogram_CT_C[i, j] = torch.poisson(sinogramCT[i, j])
    # to density
    sinogram_CT_D = sinogram_CT_C / IO
    siongram_out = - max_sinogram * torch.log(sinogram_CT_D)
    return siongram_out


def add_poisson_to_sinogram_torch_fast(sinogram, IO, seed=None):
    if seed is not None:
        torch.manual_seed(seed)

    max_sinogram = sinogram.max()
    sinogramRawScaled = sinogram / max_sinogram.max()
    sinogramCT = IO * torch.exp(-sinogramRawScaled)
    sinogramCT = torch.clamp(sinogramCT, min=0.0)
    sinogram_CT_C = torch.poisson(sinogramCT)
    sinogram_CT_D = sinogram_CT_C / IO
    siongram_out = - max_sinogram * torch.log(sinogram_CT_D)
    return siongram_out


# add guassian nose
def add_gaussian_to_sinogram_checktype(sinogram, sigma=25.0, seed=None):
    assert isinstance(sinogram, torch.Tensor)
    if seed is not None:
        torch.manual_seed(seed)
    dtype = sinogram.dtype

    if not sinogram.is_floating_point():
        sinogram = sinogram.to(torch.float32)
    noisy_sinogram = sinogram + sigma * torch.randn_like(sinogram)

    if noisy_sinogram.dtype != dtype:
        noisy_sinogram = noisy_sinogram.to(dtype)
    return noisy_sinogram

def add_gaussian_to_sinogram_torch(sinogram, sigma=25.0, seed=None):
    assert isinstance(sinogram, torch.Tensor)
    if seed is not None:
        torch.manual_seed(seed)
    noisy_sinogram = sinogram + sigma * torch.randn_like(sinogram)
    return noisy_sinogram



if __name__ == '__main__':
    pass