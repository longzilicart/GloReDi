# Basic Module for Sparse-view CT by V0 version longzili 

# including:
    # Basic_Sparse_Model: basic moduel for spaser_view CT, support projection and forward-back projection, and adding noise
    # SparseCT_Conv_Net: basic moduel for any image domain network
    # SparseCT_Teacher_Net: basic module for distillation framework
    # SparseCT_Dudo_Net: basic module for dual-domain network






import torch
import torch.nn as nn
import numpy as np
import random
from torch_radon import Radon, RadonFanbeam
from torch_radon.solvers import cg
from torch_radon.shearlet import ShearletTransform
import torch_radon
from utils.ct_tools import *


class Basic_Radon_Param:
    # basic projection parameters for sparse-view CT
    angle = 720
    source_distance = 1075 # fanbeam distance
    d_count = 672 # detector num , basic setting
    spacing = 0.08 # 
    img_size = 256
    imPixScale = 512 / img_size * spacing

class Sparse_Radon_Angle_Num:
    angle18 = 18
    angle36 = 36
    angle72 = 72









# ======= Basic single domain Sparse Model ======
# by default, input is a valid miu_ct, sparse_model will generate [Spaser_CT and groundtruth]
class Basic_Sparse_Model(nn.Module):
    '''Sparse network Basic Model
    Basic transform
    input:
        a miu_ct tensor
        t -> full radon as groundtruth
        t -> sparse radon as func
    '''
    def __init__(self, sparse_angle=None, poisson_rate=-1, gaussian_rate=-1):
        super(Basic_Sparse_Model, self).__init__()
        basic_radon_param = Basic_Radon_Param()
        self.full_angle = basic_radon_param.angle
        self.source_distance = basic_radon_param.source_distance
        self.d_count = basic_radon_param.d_count
        self.img_size = basic_radon_param.img_size
        self.imPixScale = basic_radon_param.imPixScale
        if sparse_angle is None:
            raise ValueError('sparse_angle not provided, need an int value, 18 for example')
        else:
            self.sparse_angle = sparse_angle
            print('full_angle: {} // sparse_angle = {}'.format(self.full_angle, self.sparse_angle))
        # self.sparse_net = sparse_net
        self.poisson_rate = poisson_rate
        self.gaussian_rate = gaussian_rate
    
    def forward(self, miu_ct):
        sparse_ct_miu, gt_ct_miu = self.sparse_pre_processing(miu_ct)
        output_ct_miu = self.sparse_net(sparse_ct_miu,)
        return sparse_ct_miu, output_ct_miu, gt_ct_miu

    @torch.no_grad()
    def sparse_pre_processing(self, miu_ct, angle_bias = 0):
        '''provide sparse ct image and groundtruth'''
        # miu_ct 不需要可导
        full_sinogram = self.image_radon(miu_ct)
        sparse_sinogram = self.image_radon(miu_ct, self.sparse_angle, angle_bias)
        sparse_sinogram = self.add_noise(sparse_sinogram)
        gt_miu_ct = self.radon(full_sinogram,)
        sparse_miu_ct = self.radon(sparse_sinogram, self.sparse_angle, angle_bias)
        return sparse_miu_ct, gt_miu_ct

    # ------------ basic radon function ----------------
    def radon(self, sinogram, angle = None, angle_bias = 0):
        '''sinogram to ct image'''
        sinogram = sinogram / self.imPixScale
        if angle is None:
            angle = self.full_angle
        else:
            angle = angle
        angles = np.linspace(0, np.pi*2, angle, endpoint=False)
        # angles with start angle bias
        if angle_bias != 0:
            # assert angle_bias < np.pi*2/angle
            angles = self.angle_list_w_bias(angles, angle_bias)
        radon = RadonFanbeam(self.img_size, angles, self.source_distance, det_count=self.d_count,)
        ma_rotate = sinogram
        filter_sin = radon.filter_sinogram(ma_rotate, "ram-lak")
        back_proj = radon.backprojection(filter_sin) 
        return back_proj 
    
    def image_radon(self, ct_image, angle = None, angle_bias = 0):
        '''ct image to sinogram'''
        ct_image = ct_image * self.imPixScale
        if angle is None:
            angle = self.full_angle
        else:
            angle = angle 
        angles = np.linspace(0, np.pi*2, angle, endpoint=False)
        # angles with start angle bias
        if angle_bias != 0:
            # assert angle_bias < np.pi*2/angle
            angles = self.angle_list_w_bias(angles, angle_bias)
        radon = RadonFanbeam(self.img_size, angles, self.source_distance,det_count = self.d_count,)
        sinogram = radon.forward(ct_image)
        sinogram = sinogram
        return sinogram

    @staticmethod
    def angle_list_w_bias(angle_list, bias):
        def angle_w_bias(angle, bias):
            return angle + bias
        bias = len(angle_list)*[bias]
        angle_list = list(map(angle_w_bias, angle_list, bias))
        return angle_list

    def add_noise(self, sinogram):
        if self.poisson_rate > 0:
            sinogram = add_poisson_to_sinogram_torch_fast(sinogram, self.poisson_rate, seed=None)
        if self.gaussian_rate > 0:
            sinogram = add_gaussian_to_sinogram_torch(sinogram, sigma=self.gaussian_rate, seed=None)
        return sinogram






# ===========
class SparseCT_Conv_Net(Basic_Sparse_Model):
    '''
    Basic module for any image domain networks
    '''
    def __init__(self, sparse_net=None, sparse_angle=None, dataset_shape=None, poisson_rate=-1, gaussian_rate=-1):
        super(SparseCT_Conv_Net, self).__init__(sparse_angle=sparse_angle, poisson_rate=poisson_rate, gaussian_rate=gaussian_rate)

        # define the network
        if sparse_net is None:
            raise ValueError('sparse_net not provided, provide a network')
        self.sparse_net = sparse_net
        if dataset_shape is not None:
            self.img_size = dataset_shape

    def forward(self, miu_ct):
        sparse_ct_miu, gt_ct_miu = self.sparse_pre_processing(miu_ct)
        output_ct_miu = self.sparse_net(sparse_ct_miu,)
        return sparse_ct_miu, output_ct_miu, gt_ct_miu


class SparseCT_Teacher_Net(Basic_Sparse_Model):
    '''
    Basic module for distillation framework
    '''
    def __init__(self, sparse_net, 
                    sparse_angle = 18,
                    teacher_angle = 180,
                    dataset_shape = 512,
                    poisson_rate = -1,
                    gaussian_rate = -1):
        super(SparseCT_Teacher_Net, self).__init__(sparse_angle=sparse_angle, poisson_rate=poisson_rate, gaussian_rate=gaussian_rate)
        # basic hyperparameter
        self.img_size = dataset_shape
        self.sparse_net = sparse_net
        # multi angle
        self.sparse_angle = sparse_angle
        self.teacher_angle = teacher_angle
        print('full_angle: {} // sparse_angle = {} // teacher_angle = {}'.format(self.full_angle, self.sparse_angle, self.teacher_angle))

    def forward(self, sparse_ct_miu):
        # output a container
        output = self.sparse_net(sparse_ct_miu)
        return output

    def sparse_pre_processing_w_multiview(self, miu_ct):
        # different noise for teacher and student, it make the task harder
        sinogram = self.image_radon(miu_ct, self.sparse_angle)        
        sinogram = self.add_noise(sinogram)
        sparse_miu_ct = self.radon(sinogram, self.sparse_angle)
        sinogram = self.image_radon(miu_ct, self.teacher_angle)
        sinogram = self.add_noise(sinogram)        
        enhance_miu_ct = self.radon(sinogram, self.teacher_angle)
        sinogram = self.image_radon(miu_ct, self.full_angle)
        gt_miu_ct = self.radon(sinogram, self.full_angle)
        return sparse_miu_ct, enhance_miu_ct, gt_miu_ct



class SparseCT_Dudo_Net(Basic_Sparse_Model):
    '''Basic Model for dual-domain netowrk
    intro: 
        Basic Module for dual-domain network
    preprocessing: 
        sparse_sinogram(sparse axis = 0, no reduce), sparse_CT
    process:
        sino_out = sino_net(sparse_sinogram)
        sino_img = RIL(sino_out)
        y = img_net(sino_img, sparse_CT) # if residual + Sparse_CT
    args:
        the same
    '''
    def __init__(self, sparse_net=None, sparse_angle=None, dataset_shape=None, poisson_rate=-1, gaussian_rate=-1):
        super(SparseCT_Dudo_Net, self).__init__(sparse_angle=sparse_angle, poisson_rate=poisson_rate, gaussian_rate=gaussian_rate)
        if sparse_angle is None:
            raise ValueError('sparse_net not provided, provide a network')
        self.sparse_net = sparse_net
        if dataset_shape is not None:
            self.img_size = dataset_shape

    def sparsedudo_pre_processing(self, miu_ct, return_sinomask = False):
        # about 0.5s, 1g memory
        sinogram_full_t = self.image_radon(miu_ct)
        gt_ct = self.radon(sinogram_full_t)
        # label无噪声
        sinogram_full_t = self.add_noise(sinogram_full_t)

        sparse_sinogram_mask = self.gen_sparse_sinogram_mask(self.sparse_angle, self.full_angle)
        sparse_sinogram = (sparse_sinogram_mask * sinogram_full_t.permute(0,1,3,2)).permute(0,1,3,2)
        sparse_sinogram_reduce = sinogram_full_t.permute(0,1,3,2)[..., sparse_sinogram_mask != 0].permute(0,1,3,2).contiguous()

        sparse_ct = self.radon(sparse_sinogram_reduce, angle = self.sparse_angle)
        if return_sinomask:
            sparse_sinogram_mask = (sparse_sinogram_mask * torch.ones_like(sparse_sinogram.permute(0,1,3,2))).permute(0,1,3,2).contiguous()
            return sparse_sinogram, sparse_ct, sinogram_full_t, gt_ct, sparse_sinogram_mask
        else:
            return sparse_sinogram, sparse_ct, sinogram_full_t, gt_ct

    @staticmethod
    def gen_sparse_sinogram_mask(sparse_angle, full_angle,):
        singram_mask = np.arange(1, full_angle+1)
        sinogram_mask = np.ma.masked_equal(singram_mask % (full_angle//sparse_angle), 1)
        sinogram_mask = sinogram_mask.mask
        sinogram_mask = sinogram_mask.astype(np.int32)
        sinogram_mask = torch.from_numpy(sinogram_mask).cuda()
        return sinogram_mask
    
    def forward(self, x):
        y = self.sparse_net(x)
        return y








if __name__ == '__main__':
    
    pass