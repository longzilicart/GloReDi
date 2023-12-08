# calc measurements in torch

from skimage.metrics import mean_squared_error as mse
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torch
import numpy as np
from math import exp
import torch.nn.functional as F
from torch.autograd import Variable

class Image_diff():
    '''calc image measurements by skimage
    '''
    def __init__(self, image_x, image_y, showimage = False, data_range=1):
        self.img_x = image_x
        self.img_y = image_y
        self.data_range = data_range
        #print(image_x.shape,image_y.shape)

    def mse_error(self):
        self.mse_er = mse(self.img_x, self.img_y)

    def ssim_error(self):
        self.ssim = ssim(self.img_x, self.img_y, data_range=self.img_y.max() - self.img_y.min())

    def psnr_error(self):
        self.psnr_error = psnr(self.img_x, self.img_y, data_range = self.data_range) 
    
    def cal_all(self):
        return self.mse_error(), self.ssim_error(), self.psnr_error()



def compute_measure(pred, y, data_range):
    '''calc image measurements in torch
    '''
    pred_psnr = compute_PSNR(pred, y, data_range)
    pred_ssim = compute_SSIM(pred, y, data_range)
    pred_rmse = compute_RMSE(pred, y)
    pred_mse = compute_MSE(pred, y)
    return  pred_rmse, pred_psnr, pred_ssim


def compute_MSE(img1, img2):
    return ((img1 - img2) ** 2).mean()


def compute_RMSE(img1, img2):
    if type(img1) == torch.Tensor:
        return torch.sqrt(compute_MSE(img1, img2)).item()
    else:
        return np.sqrt(compute_MSE(img1, img2))


def compute_PSNR(img1, img2, data_range):
    eps = 1e-10
    mse_ = compute_MSE(img1, img2)
    if mse_ == 0:
        mse_ += eps
    if type(img1) == torch.Tensor:
        return 10 * torch.log10((data_range ** 2) / mse_).item()
    else:
        return 10 * np.log10((data_range ** 2) / mse_)


def compute_SSIM(img1, img2, data_range, window_size=11, channel=1, size_average=True):
    # referred from https://github.com/Po-Hsun-Su/pytorch-ssim
    # default window_size 11
    if len(img1.size()) == 2:
        shape_ = img1.shape[-1]
        img1 = img1.view(1,1,shape_ ,shape_ )
        img2 = img2.view(1,1,shape_ ,shape_ )
    window = create_window(window_size, channel)
    window = window.type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size//2)
    mu2 = F.conv2d(img2, window, padding=window_size//2)
    mu1_sq, mu2_sq = mu1.pow(2), mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2) - mu1_mu2

    C1, C2 = (0.01*data_range)**2, (0.03*data_range)**2
    #C1, C2 = 0.01**2, 0.03**2
    ssim_map = ((2*mu1_mu2+C1)*(2*sigma12+C2)) / ((mu1_sq+mu2_sq+C1)*(sigma1_sq+sigma2_sq+C2))
    if size_average:
        return ssim_map.mean().item()
    else:
        return ssim_map.mean(1).mean(1).mean(1).item()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window
