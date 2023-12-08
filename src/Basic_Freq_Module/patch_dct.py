# patch dct module by longzili


import sys
sys.path.append("..")
from Basic_Freq_Module.torch_dct import *

class SplitPatchDCT(nn.Module):

    def __init__(self, patch_size=8, stack_dim='stack', dct_norm = None):
        super(SplitPatchDCT, self).__init__()
        self.patch_size = patch_size
        self.stack_dim = stack_dim
        self.img_dim = None  
        self.dct_norm = dct_norm 

    def forward(self, x):
        b, c, h, w = x.size()
        self.img_dim = (h, w)
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches_dct = dct_2d(patches, norm=self.dct_norm)
        if self.stack_dim == 'c':
            return patches_dct.reshape(b, c * (h // self.patch_size * w // self.patch_size), self.patch_size, self.patch_size).contiguous()
        elif self.stack_dim == 'stack':
            return patches_dct.reshape(b, c, h // self.patch_size * w // self.patch_size, self.patch_size, self.patch_size).contiguous()
        else:
            raise NotImplementedError("only support 'c' and 'stack' for stack_dim")

    def inverse(self, patches_dct):
        h, w = self.img_dim
        patch_nums = (h * w) // (self.patch_size * self.patch_size)
        patch_dim = int(np.sqrt(patch_nums))

        if self.stack_dim == 'c':
            b, c_new, ph, pw = patches_dct.size()
            patches = idct_2d(patches_dct, norm=self.dct_norm)
            c = c_new // patch_nums
            patches = patches.reshape(b, c, patch_nums, self.patch_size, self.patch_size)
            patches = patches.permute(0, 1, 3, 4, 2).reshape(b, c, ph, pw, patch_dim, patch_dim).permute(0, 1, 4, 2, 5, 3).reshape(b, c, ph * patch_dim, pw * patch_dim)
            return patches
        elif self.stack_dim == 'stack':
            b, c, _, ph, pw = patches_dct.size()
            patches = idct_2d(patches_dct, norm=self.dct_norm)
            patches = patches.permute(0, 1, 3, 4, 2).reshape(b, c, ph, pw, patch_dim, patch_dim).permute(0, 1, 4, 2, 5, 3).reshape(b, c, ph * patch_dim, pw * patch_dim)
            return patches
        else:
            raise NotImplementedError("only support 'c' and 'stack' for stack_dim")
        

if __name__ == '__main__':

    pass