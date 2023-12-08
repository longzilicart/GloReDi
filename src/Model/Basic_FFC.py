# Fast Fourier Convolution NeurIPS 2020
# original implementation https://github.com/pkumivision/FFC/blob/main/model_zoo/ffc.py
# paper https://proceedings.neurips.cc/paper/2020/file/2fd5d41ec6cfab47e32164d5624269b1-Paper.pdf
# modified by longzili


import numpy as np
import torch
import torch. nn as nn
import torch. nn. functional as F
import torch.fft
# ---- augmentation in network, don't use this in CT tasks ----
# from kornia.geometry.transform import rotate
# from saicinpainting. training. modules. base import get_activation, BaseDiscriminator
# from saicinpainting. training. modules. spatial_transform import LearnableSpatialTransformWrapper
# from saicinpainting. training. modules. squeeze_excitation import SELayer
# from saicinpainting. utils import get_shape

class FFCSE_block(nn. Module):
    def __init__(self, channels, ratio_g):
        super(FFCSE_block, self). __init__()
        in_cg = int(channels * ratio_g)
        in_cl = channels - in_cg
        r = 16

        self. avgpool = nn. AdaptiveAvgPool2d((1, 1))
        self. conv1 = nn. Conv2d(channels, channels // r,
                               kernel_size=1, bias=True)
        self. relu1 = nn. ReLU(inplace=True)
        self. conv_a2l = None if in_cl == 0 else nn. Conv2d(
            channels // r, in_cl, kernel_size=1, bias=True)
        self. conv_a2g = None if in_cg == 0 else nn. Conv2d(
            channels // r, in_cg, kernel_size=1, bias=True)
        self. sigmoid = nn. Sigmoid()

    def forward(self, x):
        x = x if type(x) is tuple else (x, 0)
        id_l, id_g = x

        x = id_l if type(id_g) is int else torch. cat([id_l, id_g], dim=1)
        x = self. avgpool(x)
        x = self. relu1(self. conv1(x))

        x_l = 0 if self. conv_a2l is None else id_l * \
            self. sigmoid(self. conv_a2l(x))
        x_g = 0 if self. conv_a2g is None else id_g * \
            self. sigmoid(self. conv_a2g(x))
        return x_l, x_g

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        '''
        squeeze-excitation layer: channel attention
        '''
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        res = x * y.expand_as(x)
        return res

class FourierUnit(nn. Module):

    def __init__(self, in_channels, out_channels, groups=1, spatial_scale_factor=None, spatial_scale_mode='bilinear',
                 spectral_pos_encoding=False, use_se=False, se_kwargs=None, ffc3d=False, fft_norm='ortho'):
        '''
        Basic fourier uniet
        use_se: squeeze-excitation channel attention
        spatial_scale_factor:
        spatial_scale_mode:
        注意： FFCconv 使用 1*1 conv, 等同于weight on single frequency
        '''
        # bn_layer not used
        super(FourierUnit, self). __init__()
        self. groups = groups
        self. conv_layer = torch. nn. Conv2d(in_channels=in_channels * 2 + (2 if spectral_pos_encoding else 0),
                                          out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self. groups, bias=False)
        self. bn = torch. nn. BatchNorm2d(out_channels * 2)
        self. relu = torch. nn. ReLU(inplace=True)

        # squeeze and excitation block
        self. use_se = use_se
        if use_se:
            if se_kwargs is None:
                se_kwargs = {}
            self.se = SELayer(self. conv_layer. in_channels, **se_kwargs)

        self. spatial_scale_factor = spatial_scale_factor
        self. spatial_scale_mode = spatial_scale_mode
        self. spectral_pos_encoding = spectral_pos_encoding
        self.ffc3d = ffc3d
        self.fft_norm = fft_norm

    def forward(self, x):
        batch = x.shape[0]

        if self.spatial_scale_factor is not None:
            orig_size = x.shape[-2:]
            x = F.interpolate(x, scale_factor=self.spatial_scale_factor, mode=self.spatial_scale_mode, align_corners=False)

        r_size = x.size()
        # (batch, c, h, w/2+1, 2)
        fft_dim = (-3, -2, -1) if self.ffc3d else (-2, -1)
 
        # FFC convolution
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])
        # print(ffted.shape)

        if self.spectral_pos_encoding:
            height, width = ffted.shape[-2:]
            coords_vert = torch.linspace(0, 1, height)[None, None, :, None].expand(batch, 1, height, width).to(ffted)
            coords_hor = torch.linspace(0, 1, width)[None, None, None, :].expand(batch, 1, height, width).to(ffted)
            ffted = torch.cat((coords_vert, coords_hor, ffted), dim=1)

        if self.use_se:
            ffted = self.se(ffted)

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.relu(self.bn(ffted))

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        ifft_shape_slice = x.shape[-3:] if self.ffc3d else x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)

        if self.spatial_scale_factor is not None:
            output = F.interpolate(output, size=orig_size, mode=self.spatial_scale_mode, align_corners=False)

        return output


class SpectralTransform(nn.Module):
    '''spectral transformer in the paper
    包含fourier 和 local fourier。 不适用lfu, 需要手动 enable_lfu=False
    '''
    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=True, **fu_kwargs):
        # bn_layer not used
        super(SpectralTransform, self).__init__()
        self.enable_lfu = enable_lfu
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()

        self.stride = stride
        # conv1 : kernel size = 1, no bias
        # out_channels//2, 最后conv2再重建回来
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels //
                      2, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.fu = FourierUnit(
            out_channels // 2, out_channels // 2, groups, **fu_kwargs)
        if self.enable_lfu:
            self.lfu = FourierUnit(
                out_channels // 2, out_channels // 2, groups)
        self.conv2 = torch.nn.Conv2d(
            out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, x):
        # 论文中不适用downsample fourier, 都是identity
        x = self.downsample(x)
        # conv1: channel/2进FFC. 对应论文中的channel reduction
        x = self.conv1(x)
        # print(x.shape)
        output = self.fu(x)
        # print(x.shape)

        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            split_s = h // split_no
            xs = torch.cat(torch.split(
                x[:, :c // 4], split_s, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s, dim=-1),
                           dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0

        output = self.conv2(x + output + xs)
        # print(output.shape)

        return output


class FFC(nn.Module):
    '''
    ratio_gin(对应论文αin): 对应global branch有多少channel进入
    ratio_gout: 和in相同比较好
    '''
    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True,
                 padding_type='reflect', gated=False, **spectral_kwargs):
        super(FFC, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        #groups_g = 1 if groups == 1 else int(groups * ratio_gout)
        #groups_l = 1 if groups == 1 else groups - groups_g
        # print(in_cg,in_cl,out_cg,out_cl)
        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.global_in_num = in_cg
        # print('++++++', in_cl,out_cl,in_cg,out_cg)

        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(in_cl, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = module(in_cl, out_cg, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        self.convg2l = module(in_cg, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)

        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(
            in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu, **spectral_kwargs)
        # 1*1 conv
        self.gated = gated
        module = nn.Identity if in_cg == 0 or out_cl == 0 or not self.gated else nn.Conv2d
        self.gate = module(in_channels, 2, 1)

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)

        out_xl, out_xg = 0, 0
        # print(x_l.shape,x_g)
        if self.gated:
            total_input_parts = [x_l]
            if torch.is_tensor(x_g):
                total_input_parts.append(x_g)
            total_input = torch.cat(total_input_parts, dim=1)

            gates = torch.sigmoid(self.gate(total_input))
            g2l_gate, l2g_gate = gates.chunk(2, dim=1)
        else:
            g2l_gate, l2g_gate = 1, 1

        if self.ratio_gout != 1:
            out_xl = self.convl2l(x_l) + self.convg2l(x_g) * g2l_gate
        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l) * l2g_gate + self.convg2g(x_g)

        return out_xl, out_xg


class FFC_BN_ACT(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, ratio_gin=1, ratio_gout=1,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity,
                 padding_type='reflect',
                 enable_lfu=True, **kwargs):
        super(FFC_BN_ACT, self).__init__()
        self.ffc = FFC(in_channels, out_channels, kernel_size,
                       ratio_gin, ratio_gout, stride, padding, dilation,
                       groups, bias, enable_lfu, padding_type=padding_type, **kwargs)

        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        global_channels = int(out_channels * ratio_gout)
        self.bn_l = lnorm(out_channels - global_channels)
        self.bn_g = gnorm(global_channels)

        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        return x_l, x_g

class LearnableSpatialTransformWrapper(nn.Module):
    '''
    data augmentation inside the network, its useful in common CV task but not in medical image
    '''
    def __init__(self, impl, pad_coef=0.5, angle_init_range=80, train_angle=True):
        super().__init__()
        self.impl = impl
        self.angle = torch.rand(1) * angle_init_range
        if train_angle:
            self.angle = nn.Parameter(self.angle, requires_grad=True)
        self.pad_coef = pad_coef

    def forward(self, x):
        if torch.is_tensor(x):
            return self.inverse_transform(self.impl(self.transform(x)), x)
        elif isinstance(x, tuple):
            x_trans = tuple(self.transform(elem) for elem in x)
            y_trans = self.impl(x_trans)
            return tuple(self.inverse_transform(elem, orig_x) for elem, orig_x in zip(y_trans, x))
        else:
            raise ValueError(f'Unexpected input type {type(x)}')

    # data augmentation in network, a trick
    def transform(self, x):
        # transform, data augmentation
        height, width = x.shape[2:]
        pad_h, pad_w = int(height * self.pad_coef), int(width * self.pad_coef)
        x_padded = F.pad(x, [pad_w, pad_w, pad_h, pad_h], mode='reflect')
        x_padded_rotated = rotate(x_padded, angle=self.angle.to(x_padded))
        return x_padded_rotated

    def inverse_transform(self, y_padded_rotated, orig_x):
        # inverse transform
        height, width = orig_x.shape[2:]
        pad_h, pad_w = int(height * self.pad_coef), int(width * self.pad_coef)
        y_padded = rotate(y_padded_rotated, angle=-self.angle.to(y_padded_rotated))
        y_height, y_width = y_padded.shape[2:]
        y = y_padded[:, :, pad_h : y_height - pad_h, pad_w : y_width - pad_w]
        return y


import sys
sys.path.append('..')
# from Freq_residual.freq_spatial_module import *

class FFCResnetBlock(nn.Module):
    '''
    modified:
        add freq_residual option:
    '''
    def __init__(self, dim, padding_type, norm_layer, activation_layer=nn.ReLU, dilation=1,
                 spatial_transform_kwargs=None, inline=False,
                 freq_residual = False, freq_residual_learnable = False, freq_residual_maskratio = 0.5, freq_residual_mode = 'Gaussian', **conv_kwargs):
        super().__init__()
        self.conv1 = FFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                padding_type=padding_type,
                                **conv_kwargs)
        self.conv2 = FFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                padding_type=padding_type,
                                **conv_kwargs)
        
        # don't use learnalbe transform in CT image
        if spatial_transform_kwargs is not None:
            self.conv1 = LearnableSpatialTransformWrapper(self.conv1, **spatial_transform_kwargs)
            self.conv2 = LearnableSpatialTransformWrapper(self.conv2, **spatial_transform_kwargs)
        self.inline = inline
        # print(conv_kwargs)
    
        # frequency residual
        self.freq_residual = freq_residual
        if self.freq_residual:
            raise NotImplementedError
            print('use frequency residual, learnable: {}, mask_ratio: {}'.format(freq_residual_learnable, freq_residual_maskratio))
            self.F_residual = Square_DCT_Freq_torch(learnable = freq_residual_learnable, mask_ratio = freq_residual_maskratio, mode = freq_residual_mode)
            # nn_freq_atten = Square_DCT_Freq_torch(learnable=True, leanable_value_list=True, value_list = [1.0, 0.5], mask_ratio = 1.0, mode = 'Gaussian')

    def forward(self, x):
        if self.inline:
            x_l, x_g = x[:, :-self.conv1.ffc.global_in_num], x[:, -self.conv1.ffc.global_in_num:]
        else:
            x_l, x_g = x if type(x) is tuple else (x, 0)

        id_l, id_g = x_l, x_g
 
        # add frequency residual
        if self.freq_residual:
            id_l = self.F_residual(id_l)
            id_g = self.F_residual(id_g) # global frequency residual?

        # FFC_BN_ACT twice
        x_l, x_g = self. conv1((x_l, x_g))
        x_l, x_g = self. conv2((x_l, x_g))

        x_l, x_g = id_l + x_l, id_g + x_g
        out = x_l, x_g
        if self. inline:
            out = torch. cat(out, dim=1)
        return out

class ConcatTupleLayer(nn. Module):
    def forward(self, x):
        assert isinstance(x, tuple)
        x_l, x_g = x
        assert torch. is_tensor(x_l) or torch. is_tensor(x_g)
        if not torch. is_tensor(x_g):
            return x_l
        return torch. cat(x, dim=1)

def get_activation(kind='tanh'):
    if kind == 'tanh':
        return nn.Tanh()
    if kind == 'sigmoid':
        return nn.Sigmoid()
    if kind is False:
        return nn.Identity()
    raise ValueError(f'Unknown activation kind {kind}')


# LAMA GENERATOR
class FFCResNetGenerator(nn. Module):
    '''
    LAMA generator[modify: frequency residual]
    '''
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn. BatchNorm2d, padding_type='reflect', activation_layer=nn. ReLU,
                 up_norm_layer=nn. BatchNorm2d, up_activation=nn. ReLU(True),
                 init_conv_kwargs={}, downsample_conv_kwargs={}, resnet_conv_kwargs={}, spatial_transform_layers=None, spatial_transform_kwargs={},
                 add_out_act=True, max_features=1024, out_ffc=False, out_ffc_kwargs={}, freq_residual = False, freq_residual_learnable_list = [], freq_residual_maskratio_list = [], freq_residual_mode = 'Diffstride'):
        assert (n_blocks >= 0)
        super(). __init__()
        
        model = [nn. ReflectionPad2d(3),
                 FFC_BN_ACT(input_nc, ngf, kernel_size=7, padding=0, norm_layer=norm_layer,
                            activation_layer=activation_layer, **init_conv_kwargs)]

        # downsample kwargs, default ratio_gin = gout = 0, no frequency in downsample layer
        for i in range(n_downsampling):
            mult = 2 ** i
            if i == n_downsampling - 1:
                cur_conv_kwargs = dict(downsample_conv_kwargs)
                cur_conv_kwargs['ratio_gout'] = resnet_conv_kwargs. get('ratio_gin', 0)
            else:
                cur_conv_kwargs = downsample_conv_kwargs
            model += [FFC_BN_ACT(min(max_features, ngf * mult),
                                 min(max_features, ngf * mult * 2),
                                 kernel_size=3, stride=2, padding=1,
                                 norm_layer=norm_layer,
                                 activation_layer=activation_layer,
                                 **cur_conv_kwargs)]

        mult = 2 ** n_downsampling
        feats_num_bottleneck = min(max_features, ngf * mult)

        # check frequency residual option and n_blocks
        if freq_residual:
            assert n_blocks == len(freq_residual_learnable_list)
            assert n_blocks == len(freq_residual_maskratio_list)

        # resnet Fourier blocks with frequency residual, default no freq res
        for i in range(n_blocks):
            cur_resblock = FFCResnetBlock(feats_num_bottleneck, padding_type=padding_type, activation_layer=activation_layer,
                                          norm_layer=norm_layer,
                                          freq_residual=freq_residual, freq_residual_learnable = freq_residual_learnable_list[i], freq_residual_maskratio = freq_residual_maskratio_list[i],
                                          freq_residual_mode = freq_residual_mode,
                                          **resnet_conv_kwargs)
            if spatial_transform_layers is not None and i in spatial_transform_layers:
                cur_resblock = LearnableSpatialTransformWrapper(cur_resblock, **spatial_transform_kwargs)
            model += [cur_resblock]

        model += [ConcatTupleLayer()]

        # upsample
        for i in range(n_downsampling): #3
            mult = 2 ** (n_downsampling - i)
            model += [nn. ConvTranspose2d(min(max_features, ngf * mult),
                                         min(max_features, int(ngf * mult / 2)),
                                         kernel_size=3, stride=2, padding=1, output_padding=1),
                      up_norm_layer(min(max_features, int(ngf * mult / 2))),
                      up_activation]

        if out_ffc:
            model += [FFCResnetBlock(ngf, padding_type=padding_type, activation_layer=activation_layer,
                                     norm_layer=norm_layer, inline=True, **out_ffc_kwargs)]

        model += [nn. ReflectionPad2d(3),
                  nn. Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        
        self. model = nn. Sequential(*model)

    def forward(self, input):
        return self. model(input)




if __name__ == '__main__':

    init_conv_kwargs = {'ratio_gin':0,'ratio_gout':0,'enable_lfu':False}
    downsample_conv_kwargs = {'ratio_gin':0,
                              'ratio_gout':0,
                              'enable_lfu':False}
    resnet_conv_kwargs = {'ratio_gin':0.75,
                            'ratio_gout':0.75,
                            'enable_lfu':False}


    # ========== test frequency residual =========
    # del, frequency residual no use
    lama_net = FFCResNetGenerator(input_nc = 1, output_nc = 1, 
                                n_blocks=9,              
                                n_downsampling = 2,
                                add_out_act='sigmoid',
                                init_conv_kwargs=init_conv_kwargs,
                                downsample_conv_kwargs = downsample_conv_kwargs,
                                resnet_conv_kwargs=resnet_conv_kwargs,
                                freq_residual=True, freq_residual_learnable_list=9*[False], freq_residual_maskratio_list=9*[0.9]
                            )
    device = 'cuda:6'
    input = torch.randn(4,1,320,320)
    output = lama_net(input)
    print(input.shape,output.shape)