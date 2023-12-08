import torch
import torch.nn as nn
import numpy as np

class CCA_Loss(nn.Module):
    def __init__(self, lambd=1e-5):
        """
        Implementation of the loss described in the paper
        From Canonical Correlation Analysis to Self-supervised Graph Neural Networks:
        https://arxiv.org/abs/2106.12484
        :param temperature: int
        """
        super(CCA_Loss, self).__init__()
        self.lambd = lambd

    def forward(self, view1, view2):
        """
        :param view1: torch.Tensor, shape [batch_size, projection_dim]
        :param view2: torch.Tensor, shape [batch_size, projection_dim]
        :return: torch.Tensor, scalar
        """
        N = view1.shape[0]

        # z1 = (view1 - view1.mean(0)) / view1.std(0)
        # z2 = (view2 - view2.mean(0)) / view2.std(0)
        # 修改避免batch=1的情况
        if N == 1:
            z1 = view1
            z2 = view2
        else:
            z1 = (view1 - view1.mean(0)) / view1.std(0)
            z2 = (view2 - view2.mean(0)) / view2.std(0)

        c = torch.mm(z1.T, z2)
        c1 = torch.mm(z1.T, z1)
        c2 = torch.mm(z2.T, z2)

        c = c / N
        c1 = c1 / N
        c2 = c2 / N

        loss_inv = -torch.diagonal(c).sum()
        iden = torch.tensor(np.eye(c.shape[0])).to(view1.device)
        loss_dec1 = (iden - c1).pow(2).sum()
        loss_dec2 = (iden - c2).pow(2).sum()
        loss = loss_inv + self.lambd * (loss_dec1 + loss_dec2)

        return loss


if __name__ == "__main__":
    cca_loss = CCA_Loss().cuda()
    a = torch.randn(4, 1, 64, 64).cuda()
    b = torch.randn(4, 1, 64, 64).cuda()
    proj_a = torch.flatten(a, start_dim = 1)
    proj_b = torch.flatten(b, start_dim = 1)
    print(proj_a.shape)
    c = torch.mm(proj_a.T, proj_b)
    loss = cca_loss(proj_a, proj_b) * 0.0005
    print(loss)



    