import numpy as np
import tqdm
import os
import itertools
import math

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from torch.utils.data import DataLoader

# amp not support real-imag tensor, so not supported lama in torch1.7
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler

from Data_Provider.deeplesion_dataset import Deep_Lesion_Dataset
from Data_Provider.aapm_dataset import AAPM_Myo_Dataset
from Model.BasicModule import *
from Model.Basic_FFC import *
from utils.ct_tools import CT_Preprocessing
from utils.cal_acc import *
from Basic_Freq_Module.Focal_Freq_Loss import *
from Logger.longzili_logger import *
from Logger.longzili_loss_scaler import *



def myprint(message, local_rank=0):
    # print on localrank 0 if ddp
    if dist.is_initialized():
        if dist.get_rank() == local_rank:
            print(message)
    else:
        print(message)


class Trainer_Basic:
    '''Trainer Basic
    Model:
    log:
        tensorboard/wandb
    Train:
        fit function(fit, train, val wait for implement)
    '''
    def __init__(self):
        # basic value
        self.iter = 0
        self.epoch = 0
        self.cttool = CT_Preprocessing()
        # batch_size = self.opt.batch_size * self.cards
        self.loss_scaler = LossScaler(max_len=50, scale_factor=3)
        self.min_loss_scaler = LossScaler(max_len=50, scale_factor=3)

    @staticmethod
    def weights_init(m):
        # kaiming init
        classname = m.__class__.__name__                               
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)    

    def init_adam_optimizer(self, net):
        # initialize pytorch Adam optimizer // no return, self.opt, self.stepopt
        self.optimizer = torch.optim.Adam(net.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
        self.step_optimizer = StepLR(self.optimizer, step_size = self.opt.step_size, gamma=self.opt.step_gamma)

    # ---- save load checkpoint ----
    @staticmethod
    def save_checkpoint(param, path, name:str, epoch:int):
        # simply save the checkpoint by epoch
        if not os.path.exists(path):
            os.makedirs(path)
        checkpoint_path = os.path.join(path, name + '_{}_epoch.pkl'.format(epoch))
        torch.save(param, checkpoint_path)

    def save_model(self):
        '''basic save model'''
        net_param = self.net.module.state_dict() # default multicard
        checkpoint_path = os.path.join(self.opt.checkpoint_root, self.opt.checkpoint_dir)
        net_check = {
            'net_param': net_param,
            'epoch': self.epoch,
        }
        self.save_checkpoint(net_check, checkpoint_path, self.opt.checkpoint_dir + '-net', self.epoch)

    def save_opt(self):
        '''basic save opt'''
        checkpoint_path = os.path.join(self.opt.checkpoint_root, self.opt.checkpoint_dir)
        opt_param = self.optimizer.state_dict()
        step_opt_param = self.step_optimizer.state_dict() # if step opt is True
        opt_check = {
            'optimizer': opt_param,
            'step_optimizer': step_opt_param,
            'epoch' : self.epoch,
        }
        self.save_checkpoint(opt_check, checkpoint_path, self.opt.checkpoint_dir +'-opt', self.epoch)

    def load_model(self, strict=True):
        '''basic load model'''
        net_checkpath = self.opt.net_checkpath
        net_checkpoint = torch.load(net_checkpath, map_location = 'cpu')
        self.net.load_state_dict(net_checkpoint['net_param'], strict=strict) # strict = False
        myprint('finish loading network')

    def load_opt(self):
        '''basic load opt'''
        opt_checkpath = self.opt.opt_checkpath
        opt_checkpoint = torch.load(opt_checkpath, map_location = 'cpu')
        self.optimizer.load_state_dict(opt_checkpoint['optimizer'])
        self.step_optimizer.load_state_dict(opt_checkpoint['step_optimizer'])
        self.epoch = opt_checkpoint['epoch']
        myprint('finish loading opt')

    def resume(self):
        '''resume training
        '''
        if self.opt.net_checkpath is not None:
            self.load_model()
            myprint('finish loading model')
        else:
            raise ValueError('opt.net_checkpath not provided')

    def resume_opt(self,):
        if self.opt.resume_opt is True and self.opt.opt_checkpath is not None:
            self.load_opt()
            myprint('finish loading optimizer')
        else:
            myprint('opt.opt_checkpath not provided')     

    @staticmethod
    def load_state_dict_byhand(net, checkpoint, net_depth = [1,2], checkpoint_depth = [2,3]):
        def search_param_by_name(name, checkpoint_keys, checkpoint_num):
            for check_num in range(checkpoint_num, len(checkpoint_keys)):
                # for i in [3, 4]:
                for i in net_depth:
                    net_name = ".".join(name.split('.')[i:])
                    for j in checkpoint_depth:
                        check_name = ".".join(checkpoint_keys[check_num].split('.')[j:])
                        if net_name == check_name:
                            # print(check_num)
                            return check_num
            return -1

        checkpoint_keys = list(checkpoint.keys())
        checkpoint_num = 0
        params = net.named_parameters()
        for name, param in params:
            num = search_param_by_name(name, checkpoint_keys, checkpoint_num)
            if num == -1:
                pass
            else:
                checkpoint_num = num
                myprint('loading {} from {}'.format(name, checkpoint_keys[checkpoint_num]))
                param.data = checkpoint[checkpoint_keys[checkpoint_num]] # order dict
        myprint('finish loading')
    
    def resume_teacher(self,):
        '''resume teacher network by hand'''
        if self.opt.student_checkpath is not None:
            # load student network
            net_checkpath = self.opt.student_checkpath
            net_checkpoint = torch.load(net_checkpath, map_location = 'cpu')
            self.load_state_dict_byhand(self.net.encoder1, net_checkpoint['net_param'])
            if self.opt.teacher_checkpath is not None:
                net_checkpath = self.opt.teacher_checkpath
                net_checkpoint = torch.load(net_checkpath, map_location = 'cpu')
                self.load_state_dict_byhand(self.net.encoder2, net_checkpoint['net_param'])
                self.load_state_dict_byhand(self.net.decoder, net_checkpoint['net_param']) # load teacher decoder
            myprint('finish loading model')
        else:
            raise ValueError('opt.student_checkpath not provided')

    def resume_from_selfrecon(self,):
        '''resume from pretrained selfrecon-debug only'''
        if self.opt.net_checkpath is not None:
            net_checkpath = self.opt.net_checkpath
            net_checkpoint = torch.load(net_checkpath, map_location = 'cpu')
            self.load_state_dict_byhand(self.net, net_checkpoint['net_param'], net_depth = [2,3])
        else:
            raise ValueError('opt.net_checkpath not provided')


    # ---- common loss function ----
    def pixel_loss(self, input, target, mode = 'l1'):
        assert mode in ['l1', 'sml1', 'l2']
        if mode == 'l1':
            L1loss = torch.nn.L1Loss(reduction = 'mean')
            loss = L1loss(input, target)        
        elif mode == 'sml1':
            smL1loss = torch.nn.SmoothL1Loss(reduction = 'mean')
            loss = smL1loss(input, target)
        elif mode == 'l2':
            mse_loss = torch.nn.MSELoss(reduction = 'mean')
            loss = mse_loss(input, target)
        else:
            raise ValueError('pixel_loss error: mode not in [l1,sml1,l2]')
        # return self.loss_scaler(loss)
        return loss

    def scale_pixel_loss(self, input, target, mode = 'l1'):
        assert mode in ['l1', 'sml1', 'l2']
        if mode == 'l1':
            L1loss = torch.nn.L1Loss(reduction = 'mean')
            loss = L1loss(input, target)        
        elif mode == 'sml1':
            smL1loss = torch.nn.SmoothL1Loss(reduction = 'mean')
            loss = smL1loss(input, target)
        elif mode == 'l2':
            mse_loss = torch.nn.MSELoss(reduction = 'mean')
            loss = mse_loss(input, target)
        else:
            raise ValueError('pixel_loss error: mode not in [l1,sml1,l2]')
        # return loss
        return self.loss_scaler(loss)

    def scale_pixel_loss_student(self, input, target, mode = 'l1'):
        assert mode in ['l1', 'sml1', 'l2']
        if mode == 'l1':
            L1loss = torch.nn.L1Loss(reduction = 'mean')
            loss = L1loss(input, target)        
        elif mode == 'sml1':
            smL1loss = torch.nn.SmoothL1Loss(reduction = 'mean')
            loss = smL1loss(input, target)
        elif mode == 'l2':
            mse_loss = torch.nn.MSELoss(reduction = 'mean')
            loss = mse_loss(input, target)
        else:
            raise ValueError('pixel_loss error: mode not in [l1,sml1,l2]')
        # return loss
        return self.min_loss_scaler(loss)

    @staticmethod
    def info_loss(input, target, mode = 'crossentropy'):
        assert mode in ['crossentropy', 'mine']
        if mode == 'crossentropy':
            infoloss = torch.nn.CrossEntropyLoss()
            loss = infoloss(input, target)
        elif mode == 'mine': 
            pass 
        # calc loss
        return loss
    
    @staticmethod
    def prob_loss(input, target, mode = 'kl'):
        assert mode in ['kl']
        if mode == 'kl':
            loss = F.kl_div(input.softmax(dim=-1).log(), target.softmax(dim=-1), reduction='sum')
        return loss

    @staticmethod
    def mse_loss(input, target):
        return F.mse_loss(input, target)

    @staticmethod
    def _to_freq_domain(x, mode = 'dct', norm = 'ortho'):
        '''to frequency domain: DCT/FFT/Wavelet'''
        if mode == 'dct':
            y = dct_2d(x, norm = norm)        
        return y
    
    @staticmethod
    def _to_image_domain(x, mode = 'dct'):
        pass

    def laplace_loss(self, ):
        pass
    
    def edge_loss(self, ):
        pass

    @staticmethod
    def cosin_loss(input, target, mode = 'cosine_sim'):
        focal = 1
        assert mode in ['cosine_sim',]
        if mode == 'cosine_sim':
            return torch.pow((1.0 - torch.cosine_similarity(input, target, dim=1)),focal).mean() 

    @staticmethod
    def focal_cosin_loss(input, target, mode = 'cosine_sim', focal = 1):
        assert mode in ['cosine_sim',]
        if mode == 'cosine_sim':
            if focal == 1:
                return (1.0 - torch.cosine_similarity(input, target, dim=1)).mean()
            elif focal != 1:
                return torch.pow((1.0 - torch.cosine_similarity(input, target, dim=1)),focal).mean() 

    # basic Sparse_CT accuracy in a CT window
    def cal_miu_ct_acc_by_window(self, miu_input, miu_target):
        hu_input = self.cttool.window_transform(self.cttool.miu2HU(miu_input))
        hu_target = self.cttool.window_transform(self.cttool.miu2HU(miu_target))
        # data_range of normalized HU img
        data_range = 1
        rmse, psnr, ssim = compute_measure(hu_input, hu_target, data_range)
        return rmse, psnr, ssim

    def to_hu(self, x):
        return self.cttool.window_transform(self.cttool.miu2HU(x))   

    # reduce function
    def reduce_value(self, value, average=True):
        world_size = torch.distributed.get_world_size()
        if world_size < 2:  # single GPU
            return value
        if not torch.is_tensor(value):
            value = torch.tensor(value)
        if not value.is_cuda:
            value = value.cuda(self.opt.local_rank)
        with torch.no_grad():
            dist.all_reduce(value)   # get reduce value
            if average:
                value = value.float()
                value /= world_size
        return value.cpu()

    def reduce_loss(self, loss, average=True):
        return self.reduce_value(loss, average=average)

    # ---- training fucntion ----
    def fit():
        raise ValueError('function fit() not implemented')

    def train():
        pass

    def val():
        pass
