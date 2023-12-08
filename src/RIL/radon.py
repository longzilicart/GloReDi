import sys
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch_radon import Radon, RadonFanbeam
from torch_radon.solvers import cg
from torch_radon.shearlet import ShearletTransform
import torch_radon

# ====== Basic radon func in SparseCT ======
# ---------------- intro ----------------
# 1. torch_radon input should be **valid miu_ct** (!!!!)
# 2. Radon_Param_Basic.start_angle_bias < 360/angle
# 3. 

class Radon_Param_Basic:
    angle = 720
    source_distance = 1075
    d_count = 672
    spacing = 0.08  # no use in our sparse CT radon implement
    img_size = 512 # default size
    start_angle_bias = np.pi * 2 / angle #
    imPixScale = 512 / img_size * spacing

radon_param_basic = Radon_Param_Basic()



def radon(sinogram, sparse_angle = None, angle_bias = 0, img_size=None):
    ''' sinogram to CT_image
    intro: 
        project sinogram to CT image by ram-lak
    args:
        sinogram:
        sparse_angle: -> int
        angle_bias: -> float, calc by angle_list_w_bias()
    return:
        CT_image in miu = back_proj
    '''
    sinogram = sinogram / Radon_Param_Basic.imPixScale
    if sparse_angle is None:
        angle = radon_param_basic.angle
    else:
        angle = sparse_angle
    d_count = radon_param_basic.d_count
    angles = np.linspace(0, np.pi*2, angle, endpoint=False)
    
    # if angle_bias, angles with start angle bias
    if angle_bias != 0:
        # assert angle_bias < np.pi*2/angle
        angles = angle_list_w_bias(angles, angle_bias)
    
    if img_size is None:
        img_size = radon_param_basic.img_size

    source_distance = radon_param_basic.source_distance
    radon = RadonFanbeam(img_size,angles,source_distance,det_count = d_count,)

    ma_rotate = sinogram
    filter_sin = radon.filter_sinogram(ma_rotate, "ram-lak")
    back_proj = radon.backprojection(filter_sin) 
    return back_proj 

def image_radon(image, sparse_angle = None, angle_bias = 0, img_size=None):
    ''' CT_image to sinogram
    intro: 
        construct CT image to sinogram
    args:
        image: CT image in valid miu
        sparse_angle: -> int
        angle_bias: -> float, calc by angle_list_w_bias()
    return:
        CT_image in miu = back_proj
    '''
    image = image * Radon_Param_Basic.imPixScale
    # ct_image to sinogram
    if sparse_angle is None:
        angle = radon_param_basic.angle
    else:
        angle = sparse_angle
    d_count = radon_param_basic.d_count
    angles = np.linspace(0, np.pi*2, angle, endpoint=False)
    
    # angles with start angle bias
    if angle_bias != 0:
        # assert angle_bias < np.pi*2/angle
        angles = angle_list_w_bias(angles, angle_bias)
    if img_size is None:
        img_size = radon_param_basic.img_size

    source_distance = radon_param_basic.source_distance
    radon = RadonFanbeam(img_size,angles,source_distance,det_count = d_count,)
    sinogram = radon.forward(image)

    sinogram = sinogram
    return sinogram


# ------------ basic radon function ----------------
def angle_list_w_bias(angle_list, bias):
    def angle_w_bias(angle, bias):
        return angle + bias
    bias_list = len(angle_list) * [bias]
    angle_list = list(map(angle_w_bias, angle_list, bias_list))
    return angle_list



if __name__ == '__main__':
    pass