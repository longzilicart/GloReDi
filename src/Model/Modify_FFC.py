# FFC encoder and decoder basic on Fourier convolution
# including basic encoder-decoder for GloReDi and Ushape encoder-decoder for GloReDi++



import numpy as np
import torch
import torch. nn as nn
import torch. nn. functional as F
import torch.fft

import sys
sys.path.append("..")
from Model.Basic_FFC import *


class Encoder_Down(nn.Module):
    def __init__(self, n_downsampling, ngf=64, max_features=1024,
                norm_layer=nn.BatchNorm2d,activation_layer=nn.ReLU, resnet_conv_kwargs={}, downsample_conv_kwargs={}):
        super(Encoder_Down, self).__init__()
        
        model = []
        for i in range(n_downsampling):
            mult = 2 ** i
            if i == n_downsampling - 1:
                cur_conv_kwargs = dict(downsample_conv_kwargs)
                cur_conv_kwargs['ratio_gout'] = resnet_conv_kwargs.get('ratio_gin', 0)
            else:
                cur_conv_kwargs = downsample_conv_kwargs
            model += [FFC_BN_ACT(min(max_features, ngf * mult),
                                 min(max_features, ngf * mult * 2),
                                 kernel_size=3, stride=2, padding=1,
                                 norm_layer=norm_layer,
                                 activation_layer=activation_layer,
                                 **cur_conv_kwargs)]
        self.down = nn.Sequential(*model)
        
    def forward(self,x):
        return self.down(x)


class Decoder_Up(nn.Module):
    def __init__(self,n_downsampling,max_features=1024,ngf=64,
                up_norm_layer=nn.BatchNorm2d, up_activation=nn.ReLU(True),):
        super(Decoder_Up, self).__init__()

        model = []
        for i in range(n_downsampling): #3
            mult = 2 ** (n_downsampling - i)
            model += [nn. ConvTranspose2d(min(max_features, ngf * mult),
                                         min(max_features, int(ngf * mult / 2)),
                                         kernel_size=3, stride=2, padding=1, output_padding=1),
                      up_norm_layer(min(max_features, int(ngf * mult / 2))),
                      up_activation]
        self.up = nn.Sequential(*model)

    def forward(self,x):
        return self.up(x)

class FFC_Bottle(nn.Module):
    ''''''
    def __init__(self,n_downsampling, max_features=1024, ngf=64, 
                n_blocks=6, padding_type='reflect', inline = False,
                activation_layer='ReLU', norm_layer=nn.BatchNorm2d, spatial_transform_layers=None, spatial_transform_kwargs=None,resnet_conv_kwargs=None):
        super(FFC_Bottle, self).__init__()
        
        model = []
        mult = 2 ** n_downsampling
        feats_num_bottleneck = min(max_features, ngf * mult)
        for i in range(n_blocks):
            cur_resblock = FFCResnetBlock(feats_num_bottleneck,             
                                        padding_type=padding_type, activation_layer=activation_layer,
                                        inline = inline, norm_layer=norm_layer, **resnet_conv_kwargs)
            if spatial_transform_layers is not None and i in spatial_transform_layers:
                cur_resblock = LearnableSpatialTransformWrapper(cur_resblock, **spatial_transform_kwargs)
            model += [cur_resblock]
        self.ffc_resnet = nn.Sequential(*model)

    def forward(self,x):
        return self.ffc_resnet(x)

class Eplicit_FFC_Bottle(nn.Module):
    def __init__(self, n_downsampling, max_features=1024, ngf=64,
                n_blocks=6, padding_type='reflect',
                activation_layer='ReLU', norm_layer=nn.BatchNorm2d,resnet_conv_kwargs=None):
        super(FFC_Bottle, self).__init__()
        mult = 2 ** n_downsampling
        feats_num_bottleneck = min(max_features, ngf * mult)
        
        # 【mannel resblock】 n_blocks
        self.ffc_res1 = FFCResnetBlock(feats_num_bottleneck,
                    padding_type=padding_type, activation_layer=activation_layer,
                    norm_layer=norm_layer, **resnet_conv_kwargs)
        self.ffc_res2 = FFCResnetBlock(feats_num_bottleneck,
                    padding_type=padding_type, activation_layer=activation_layer,
                    norm_layer=norm_layer, **resnet_conv_kwargs)
        self.ffc_res3 = FFCResnetBlock(feats_num_bottleneck,
                    padding_type=padding_type, activation_layer=activation_layer,
                    norm_layer=norm_layer, **resnet_conv_kwargs)
        self.ffc_res4 = FFCResnetBlock(feats_num_bottleneck,
                    padding_type=padding_type, activation_layer=activation_layer,
                    norm_layer=norm_layer, **resnet_conv_kwargs)
        # self.ffc_res5 = FFCResnetBlock(feats_num_bottleneck,
        #             padding_type=padding_type, activation_layer=activation_layer,
        #             norm_layer=norm_layer, **resnet_conv_kwargs)
    def forward(self, x):
        x1 = self.ffc_res1(x)
        x2 = self.ffc_res2(x1)
        x3 = self.ffc_res3(x2)
        x4 = self.ffc_res4(x3)
        y = x4
        return y


class FFC_Encoder(nn. Module):
    '''
    [support odd input]
    '''
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=6, 
                norm_layer=nn.BatchNorm2d, padding_type='reflect', activation_layer=nn.ReLU,
                init_conv_kwargs={}, downsample_conv_kwargs={}, resnet_conv_kwargs={},
                add_out_act=True, max_features=1024, ):
        assert (n_blocks >= 0)
        super(). __init__()
        self.reflect = nn.Sequential(
                nn. ReflectionPad2d(3),
                FFC_BN_ACT(input_nc, ngf, kernel_size=7, padding=0, norm_layer=norm_layer,
                            activation_layer=activation_layer, **init_conv_kwargs))
        # downsample
        self.downsample = Encoder_Down(n_downsampling, ngf, max_features, 
                    norm_layer, activation_layer, 
                    resnet_conv_kwargs=resnet_conv_kwargs, downsample_conv_kwargs=downsample_conv_kwargs)
        
        # ffc resblock
        self.ffc_bottle = FFC_Bottle(n_downsampling, ngf = ngf, max_features = max_features,
                 n_blocks = n_blocks, padding_type = padding_type, activation_layer =  activation_layer, norm_layer = norm_layer,
                resnet_conv_kwargs = resnet_conv_kwargs)
        self.cattuple_layer = ConcatTupleLayer()

    def forward(self, x):
        x = self.reflect(x)
        x = self.downsample(x)
        x = self.ffc_bottle(x)
        latent_x = self.cattuple_layer(x)
        return latent_x

    
class FFC_Decoder(nn. Module):
    '''
    FFC decoder
    '''
    def __init__(self, input_nc, output_nc, n_downsampling=3, n_blocks=6, 
                ngf=64, max_features=1024, 
                norm_layer=nn.BatchNorm2d, padding_type='reflect', activation_layer=nn.ReLU, up_norm_layer=nn.BatchNorm2d, up_activation=nn.ReLU(True),
                resnet_conv_kwargs={}, add_out_act=True, out_ffc=False, out_ffc_kwargs={}):
        assert (n_blocks >= 0)
        super(). __init__()

        # decoder ffc resblock [latent_x -> decoder ffc_block]
        self. totuple_layer =  ToTupleLayer(resnet_conv_kwargs.get('ratio_gin', 0))
        self.ffc_bottle = FFC_Bottle(n_downsampling, max_features = max_features,
                ngf = ngf, n_blocks = n_blocks, padding_type = padding_type, activation_layer =  activation_layer, norm_layer = norm_layer, inline = False, resnet_conv_kwargs = resnet_conv_kwargs)

        # upsample
        self.cattuple_layer = ConcatTupleLayer()
        self.upsample = Decoder_Up(n_downsampling,max_features,ngf,
                                    up_norm_layer, up_activation)
        # final_process
        model = []
        if out_ffc:
            model += [FFCResnetBlock(ngf, padding_type=padding_type, activation_layer=activation_layer,
                                    norm_layer=norm_layer, inline=True, **out_ffc_kwargs)]
        model += [nn. ReflectionPad2d(3),
                nn. Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        if add_out_act:
            model. append(get_activation('sigmoid' if add_out_act is True else add_out_act))
        self. final_process = nn. Sequential(*model)

    def forward(self, x):
        x = self.totuple_layer(x)
        x = self.ffc_bottle(x)
        x = self.cattuple_layer(x)
        x = self.upsample(x)
        y = self.final_process(x)
        return y


class ToTupleLayer(nn. Module):
    def __init__(self, ratio_gin,):
        super().__init__()
        self.ratio_gin = ratio_gin
    def forward(self, x):
        assert torch.is_tensor(x)
        _, c, _, _ = x.size()
        if self.ratio_gin != 0:
            x_l, x_g = x[:, : -int(self.ratio_gin * c)], x[:, : int(self.ratio_gin * c)]
        else:
            x_l, x_g = x, 0
        # print(x_l.shape, x_g.shape)
        out = x_l, x_g
        return out






# =========== GloReDi plus ============
# =========== GloReDi plus ============
# =========== GloReDi plus ============
# =========== GloReDi plus ============







if __name__ == '__main__':
    pass