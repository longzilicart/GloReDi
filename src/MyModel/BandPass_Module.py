import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("..")
from utils.SCL_loss import SupervisedContrastiveLoss
from utils.CCA_loss import CCA_Loss
from utils.gen_mask import *
from Basic_Freq_Module.patch_dct import *
from Basic_Freq_Module.Freq_norm import *



class BandPass_SCL_Module(nn.Module):
    '''module for bandpass mask selection v0 version
    kwargs:
        scl_mode: ['spatial', 'dct', 'patchdct']
        scl_memory_length: depend on cuda memory
        scl_memory_device: support only cuda
        scl_temperature: [0.07, 0.1, 0.2]
        scl_low_ratio: low frequency of the bandpass filter
        scl_high_ratio: high frequency of the bandpass filter
        scl_save_mode: contrastive loss based on teacher or student feature
        learnable: whether the bandpass filter is learnable
        learnable_mode: 
    '''
    def __init__(self,
                scl_mode = 'dct',
                scl_memory_length = 100,
                scl_memory_device = 'cuda',
                scl_temperature = 0.1,
                scl_low_ratio = 0.5,
                scl_high_ratio = -1,
                scl_save_mode = 'student',
                scl_learnable = False,
                scl_learnable_mode = None,
                ):
        super().__init__()

        # 【2】SCL loss related
        self.memory_length = scl_memory_length
        self.memory_device = scl_memory_device
        self.memory_bank = None
        self.memory_static = 0 
        if scl_mode is not None:
            assert scl_mode in ['spatial', 'dct', 'patchdct'] 
        self.scl_mode = scl_mode

        self.scl_learnable = scl_learnable
        if scl_learnable:
            self.scl_low_ratio = nn.Parameter(torch.tensor(scl_low_ratio), requires_grad=True)
            self.scl_high_ratio = nn.Parameter(torch.tensor(scl_high_ratio), requires_grad=True)
        else: # not learnable
            self.scl_low_ratio = scl_low_ratio # low frequency 
            self.scl_high_ratio = scl_high_ratio # high frequency

        self.SCL_Loss = SupervisedContrastiveLoss(temperature = scl_temperature,)
        assert scl_save_mode in ['student', 'teacher']
        self.scl_save_mode = scl_save_mode
        self.dct_mask = None

        # patch DCT layer, 
        self.patchdct_layer = SplitPatchDCT(patch_size = 64, stack_dim='c', dct_norm = None) 

        # self.register_buffer(f'{"memory_bank"}', torch.empty(self.memory_length, self.memory_dim))

    # === SCL_Loss on DCT===
    def get_scl_loss(self, student_features, teacher_features, scl_mode=None):
        if scl_mode is not None:
            self.scl_mode = scl_mode
        if self.scl_mode == 'spatial':
            scl_loss = self.calc_scl_loss_on_spatial(student_features, teacher_features)
        elif self.scl_mode == 'dct':
            if self.scl_learnable:
                scl_loss = self.calc_scl_loss_on_dct_learn(student_features, teacher_features)
            else:
                scl_loss = self.calc_scl_loss_on_dct(student_features, teacher_features)
        return scl_loss

    def calc_scl_loss_on_dct(self, student_feats, teacher_feats):
            dct_student = dct_2d(student_feats, )
            dct_teacher = dct_2d(teacher_feats, )
            # dct_student = dct_2d(student_feats, norm = 'ortho') 
            # dct_teacher = dct_2d(teacher_feats, norm = 'ortho')

            #  patchDCT option
            # dct_teacher = self.patchdct_layer(teacher_feats)
            # dct_student = self.patchdct_layer(student_feats)
            
            if self.dct_mask is None:
                self.generate_DCT2_mask(dct_student)
            

            b, c, _, _ = dct_student.size()
            dct_student = torch.masked_select(dct_student, self.dct_mask).view(b, c, -1)
            dct_teacher = torch.masked_select(dct_teacher, self.dct_mask).view(b, c, -1)
            return self.calc_scl_loss_on_spatial(dct_student, dct_teacher)

    def update_memory_bank(self, student_features, teacher_features):
        '''projection shape: (batch, dim)'''
        # project to valid dimension
        if self.scl_save_mode == 'student':
            features = student_features
        else:
            features = teacher_features
        update_feature = features.detach()
        new_projections = self.norm_projector(update_feature.to(self.memory_device))
        if self.memory_bank is None:
            self.memory_bank = [None for _ in range(self.memory_length)]
        for i in range(update_feature.size(0)):
            new_projection = new_projections[i, ...].unsqueeze(0)
            self.memory_bank.pop()
            self.memory_bank.insert(0, new_projection)
            self.memory_static = min(self.memory_static + 1, self.memory_length)

    def calc_scl_loss_on_spatial(self, student_features, teacher_features):
        if self.memory_static < self.memory_length: 
            self.update_memory_bank(student_features, teacher_features)
            return torch.tensor(0)
        b = student_features.size(0)
        student_label = [i+1 for i in range(b)]
        teacher_label = [i+1 for i in range(b)]
        memory_label = [0 for _ in range(self.memory_length)]
        labels = student_label + teacher_label + memory_label
        labels = torch.tensor(labels)
        student_projections = self.norm_projector(student_features)
        teacher_projections = self.norm_projector(teacher_features)
        if student_projections.device != self.memory_device:
            features = torch.cat([student_projections.to(self.memory_device), teacher_projections.to(self.memory_device), *self.memory_bank], dim = 0) 
        else:
            features = torch.cat((student_projections, teacher_projections, *self.memory_bank), dim = 0)
        self.update_memory_bank(student_features, teacher_features)
        return self.calc_scl_loss(features, labels)

    def calc_scl_loss(self, features, labels):
        if len(features.shape) > 2:
            projections = self.norm_projector(features)
        else:
            projections = features
        loss = self.SCL_Loss(projections, labels)
        return loss


    # ---- other tools decorator ----
    def conditional_no_grad(compute_grad=False):
        def decorator(func):
            def wrapper(*args, **kwargs):
                if compute_grad:
                    return func(*args, **kwargs)
                else:
                    with torch.no_grad():
                        return func(*args, **kwargs)
            return wrapper
        return decorator


    # if learnable compute grad
    @conditional_no_grad(compute_grad=False) 
    def generate_DCT2_mask(self, features):

        if self.scl_low_ratio < 1:
            start_ratio =  self.scl_low_ratio
        else:
            start_ratio = self.scl_low_ratio
        if self.scl_high_ratio is None:
            end_ratio = 1.0
        else:
            end_ratio = self.scl_high_ratio
            assert start_ratio < end_ratio
        int_mask = gen_DCT2_square_mask(features, start_ratio, end_ratio, center=(0,0))
        self.dct_mask = int_mask.bool().to(features.device)

    @staticmethod
    def norm_projector(features):
        projections = torch.flatten(features, start_dim = 1) #
        projections = F.normalize(projections, dim = -1) 
        return projections

    @staticmethod
    def dct_projector(features, method = 'log-scaling', eps = 1e-12):
        # normalization
        projections = dct_normalize_torch(features, method = method)
        projections = projections.clamp_min(eps)

        return projections        

    def forward(self, student_features, teacher_features, scl_mode=None):
        return self.get_scl_loss(student_features, teacher_features, scl_mode)
    
    # @torch.no_grad()
    @conditional_no_grad(compute_grad=True) 
    def generate_DCT2_mask_learn(self, features):
        if self.scl_low_ratio.item() >= self.scl_high_ratio.item():
            self.scl_high_ratio.data = self.scl_low_ratio.data + 0.1
        self.scl_low_ratio.data.clamp_(0.0, 0.9)
        self.scl_high_ratio.data.clamp_(0.1, 1.0)

        start_ratio, end_ratio = self.scl_low_ratio, self.scl_high_ratio
        int_mask = gen_DCT2_circle_mask_learn(features, start_ratio, end_ratio, center=(0,0))
        self.dct_mask = int_mask

    def calc_scl_loss_on_dct_learn(self, student_feats, teacher_feats):
        dct_student = self.patchdct_layer(student_feats)
        dct_teacher = self.patchdct_layer(teacher_feats)

        self.generate_DCT2_mask_learn(dct_student)

        b, c, _, _ = dct_student.size()
        if self.memory_static < self.memory_length:
            self.update_memory_bank_learn(dct_student, dct_teacher)
            return torch.tensor(0)
        b = dct_student.size(0)
        student_label = [i+1 for i in range(b)]
        teacher_label = [i+1 for i in range(b)]
        memory_label = [0 for _ in range(self.memory_length)]
        labels = torch.tensor(student_label + teacher_label + memory_label)
        # cat on batch
        if dct_student.device != self.memory_device:
            features = torch.cat([dct_student.to(self.memory_device), dct_teacher.to(self.memory_device), *self.memory_bank], dim = 0)
        else:
            features = torch.cat((dct_student, dct_teacher, *self.memory_bank), dim = 0)


        features = features * self.dct_mask.to(features.device)
        b, c, _, _ = features.size()
        features = torch.masked_select(features, self.dct_mask.bool()).view(b, c, -1)
        # scl projection
        projections = self.norm_projector(features)
        return self.calc_scl_loss(projections, labels)

    def update_memory_bank_learn(self, student_features, teacher_features):
        '''projection shape: (batch, dim)'''
        # project to valid dimension
        if self.scl_save_mode == 'student':
            features = student_features
        else:
            features = teacher_features
        update_feature = features.detach().to(self.memory_device)
        if self.memory_bank is None:
            self.memory_bank = [None for _ in range(self.memory_length)]
        for i in range(update_feature.size(0)):
            new_projection = update_feature[i, ...].unsqueeze(0)
            self.memory_bank.pop()
            self.memory_bank.insert(0, new_projection)
            self.memory_static = min(self.memory_static + 1, self.memory_length)









if __name__ == '__main__':
    pass




