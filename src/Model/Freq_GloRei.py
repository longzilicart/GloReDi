import torch
import torch.nn as nn
import sys
sys.path.append("..")
from utils.gen_mask import *
from Model.Basic_FFC import *
from Basic_Freq_Module.torch_dct import *
from Model.Modify_FFC import *
from Model.BasicModule import SparseCT_Teacher_Net
from MyModel.BandPass_Module import *


class GloRei_Basic(SparseCT_Teacher_Net):
    '''
    Basic Module for GloReDi
    '''
    def __init__(self, 
                    input_nc, output_nc,
                    scl_mode = 'dct',
                    scl_memory_length = 100,
                    scl_memory_device = 'cuda',
                    scl_temperature = 0.1,
                    scl_low_ratio = 0.5,
                    scl_high_ratio = -1,
                    scl_save_mode = 'student',
                    scl_learnable = True,
                    scl_learnable_mode = None,
                    global_skip = False,
                ngf=64, n_downsampling=2, e_blocks=8, d_blocks = 1,  
                norm_layer=nn.BatchNorm2d, padding_type='reflect', activation_layer=nn.ReLU,up_norm_layer=nn.BatchNorm2d, up_activation=nn.ReLU(True),
                init_conv_kwargs={}, downsample_conv_kwargs={}, resnet_conv_kwargs={},
                add_out_act=False, max_features=1024, out_ffc=False, out_ffc_kwargs={},
                simulate_kwargs = {}
                ):
        
        super(GloRei_Basic, self).__init__(None, **simulate_kwargs)
        # encoder && decoder blocks number
        print(e_blocks, d_blocks)
        assert e_blocks>0 and d_blocks>0
        self.e_blocks = e_blocks
        self.d_blocks = d_blocks
        print(f'GloRei encoder blocks:{self.e_blocks} || decoder blocks:{self.d_blocks}')

        # 【1】define the module
        self.bpSCL_module = BandPass_SCL_Module(scl_mode, scl_memory_length, scl_memory_device, scl_temperature, scl_low_ratio, scl_high_ratio, scl_save_mode, scl_learnable, scl_learnable_mode)
        self.global_skip = global_skip
        # self.patchdct_layer = SplitPatchDCT(patch_size = 64, stack_dim='c', dct_norm = None)

    def forward(self, input_list: list):
            raise NotImplementedError

    # basic train mode
    def basic_forward(self, x, encoder, decoder):
        latent = encoder(x)
        y = decoder(latent)
        if self.global_skip:
            return y + x, latent
        else:
            return y, latent

    def get_scl_loss(self, student_features, teacher_features, scl_mode=None):
        if scl_mode is not None:
            self.scl_mode = scl_mode
        scl_loss = self.bpSCL_module(student_features, teacher_features, scl_mode)
        return scl_loss

    @staticmethod
    def load_pretrain_from_lama():       
        pass



class GloRei_EMA(GloRei_Basic):
    ''' Explicit 避免传递module test
    Parallel version
        encoder1 -> decoder1
        encoder2 -> decoder2
    '''
    def __init__(self, 
                input_nc, output_nc,
                scl_mode = 'dct',
                scl_memory_length = 100,
                scl_memory_device = 'cuda',
                scl_temperature = 0.1,
                scl_low_ratio = 0.5,
                scl_high_ratio = -1,
                scl_save_mode = 'student',
                scl_learnable = True,
                scl_learnable_mode = None,
                momentum = 0.999,
                global_skip = False,
                ngf=64, n_downsampling=2, e_blocks=7, d_blocks = 2, 
                norm_layer=nn.BatchNorm2d, padding_type='reflect', activation_layer=nn.ReLU,up_norm_layer=nn.BatchNorm2d, up_activation=nn.ReLU(True),
                init_conv_kwargs={}, downsample_conv_kwargs={}, resnet_conv_kwargs={},
                add_out_act=False, max_features=1024, out_ffc=False, out_ffc_kwargs={},
                simulate_kwargs = {}):
        super(GloRei_EMA, self). __init__(input_nc, output_nc,
            scl_mode, scl_memory_length, scl_memory_device, scl_temperature, scl_low_ratio, scl_high_ratio, scl_save_mode, scl_learnable, scl_learnable_mode, global_skip,
            ngf, n_downsampling, e_blocks, d_blocks, 
            norm_layer, padding_type, activation_layer,up_norm_layer, up_activation,
            init_conv_kwargs, downsample_conv_kwargs, resnet_conv_kwargs,
            add_out_act, max_features, out_ffc, out_ffc_kwargs,
            simulate_kwargs)

        self.encoder1 = FFC_Encoder(input_nc, output_nc, ngf=ngf,
            n_downsampling=n_downsampling, n_blocks = e_blocks, max_features=max_features,
            norm_layer=norm_layer,padding_type=padding_type, activation_layer=activation_layer,
            init_conv_kwargs=init_conv_kwargs, downsample_conv_kwargs=downsample_conv_kwargs, resnet_conv_kwargs=resnet_conv_kwargs,
            add_out_act=add_out_act, )
        self.encoder2 = FFC_Encoder(input_nc, output_nc, ngf=ngf,
            n_downsampling=n_downsampling, n_blocks = e_blocks, max_features=max_features,
            norm_layer=norm_layer,padding_type=padding_type, activation_layer=activation_layer,
            init_conv_kwargs=init_conv_kwargs, downsample_conv_kwargs=downsample_conv_kwargs, resnet_conv_kwargs=resnet_conv_kwargs,
            add_out_act=add_out_act, )
        self.decoder = FFC_Decoder(input_nc, output_nc, ngf=ngf,
            n_downsampling=n_downsampling, n_blocks = d_blocks, norm_layer=norm_layer,
            padding_type=padding_type, activation_layer=activation_layer,
            up_norm_layer=up_norm_layer, up_activation=up_activation,
            resnet_conv_kwargs=resnet_conv_kwargs,
            add_out_act=add_out_act, max_features=max_features, out_ffc=out_ffc, out_ffc_kwargs=out_ffc_kwargs)    
        self.decoder2 = FFC_Decoder(input_nc, output_nc, ngf=ngf,
            n_downsampling=n_downsampling, n_blocks = d_blocks, norm_layer=norm_layer,
            padding_type=padding_type, activation_layer=activation_layer,
            up_norm_layer=up_norm_layer, up_activation=up_activation,
            resnet_conv_kwargs=resnet_conv_kwargs,
            add_out_act=add_out_act, max_features=max_features, out_ffc=out_ffc, out_ffc_kwargs=out_ffc_kwargs)        
        self.momentum = momentum # EMA_momentum
        self.init_sync_encoder_param()


    def forward(self, input_list):
        # parse input list and forward student-network or teacher network
        sparse_ct, dense_ct, stage = input_list
        assert stage in ['student', 'teacher', 'teacher_sup', 
                        'finetune','finetune_decoder']
        if stage == 'student':
            sparse_out, sparse_latent = self.basic_forward(sparse_ct, self.encoder1, self.decoder)
            return sparse_out, sparse_latent
        elif stage == 'teacher':
            # self.update_EMA_decoder_param()
            dense_out, dense_latent = self.basic_forward(dense_ct, self.encoder2, self.decoder2)
            return dense_out, dense_latent
        elif stage == 'teacher_sup':
            with torch.no_grad():
                dense_latent = self.encoder2(dense_ct)
                return dense_latent
        elif stage == 'finetune_decoder':
            with torch.no_grad():
                student_latent = self.encoder1(sparse_ct)
            y = self.decoder(student_latent)
            return y, student_latent
        elif stage == 'finetune':
            sparse_out, sparse_latent = self.basic_forward(sparse_ct, self.encoder1, self.decoder)
            return sparse_out, sparse_latent
        else:
            raise NotImplementedError
    
    @torch.no_grad()
    def update_EMA_decoder_param(self):
        for param_theta, param_fi in zip(self.decoder.parameters(),
                                         self.decoder2.parameters()):
            param_fi.data = self.momentum * param_fi.data + (
                1. - self.momentum) * param_theta.data

    @torch.no_grad()
    def init_sync_encoder_param(self):
        for param_theta, param_fi in zip(self.encoder1.parameters(),
                                         self.encoder2.parameters()):
            # param_fi.data = self.momentum * param_fi.data + (
                # 1. - self.momentum) * param_theta.data
            param_fi.data = param_theta.data










































if __name__ == '__main__':
    pass
    a = torch.randn(4, 1, 256, 256)
    b = torch.randn(4, 1, 256, 256)
    stage  = 'teacher' #'student'
    input_list = [a, b, stage]
    # define the network test only
    init_conv_kwargs = {'ratio_gin':0,'ratio_gout':0,'enable_lfu':False}
    downsample_conv_kwargs = {'ratio_gin':0, 'ratio_gout':0,'enable_lfu':False}
    resnet_conv_kwargs = {'ratio_gin':0.75,'ratio_gout':0.75,'enable_lfu':False}
    simulate_kwargs = {'sparse_angle': 18, 'teacher_angle': 72, 'dataset_shape': 512, 'poisson_rate':1e6}
    net = FreqCDL_EMA(1, 1, n_downsampling=2, e_blocks=8,d_blocks=1, add_out_act=False, init_conv_kwargs = init_conv_kwargs, downsample_conv_kwargs =    downsample_conv_kwargs, resnet_conv_kwargs = resnet_conv_kwargs,
            scl_memory_length=100, scl_memory_device='cuda', scl_temperature=0.1, scl_mode = 'dct', scl_freq_ratio=10, scl_freq_out_ratio=0.5,scl_save_mode='student', global_skip=False,
            simulate_kwargs = simulate_kwargs)
    y, latent = net(input_list)
    print(y.shape)



