import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np

class VoxelLoss(nn.Module):
    def __init__(self, alpha, beta, gamma):
        super(VoxelLoss, self).__init__()
        self.smoothl1loss = nn.SmoothL1Loss(reduction='sum')
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, reg, p_pos, pos_equal_one, neg_equal_one, targets, tag='train'):
        # reg (N * A*7 * H * W) , score (N * A * H * W)
        reg = reg.permute(0,2,3,1).contiguous()
        reg = reg.view(reg.size(0),reg.size(1),reg.size(2),-1,7) # (N * H * W * A * 7)
        targets = targets.view(targets.size(0),targets.size(1),targets.size(2),-1,7) # (N * H * W * A * 7)
        pos_equal_one_for_reg = pos_equal_one.unsqueeze(pos_equal_one.dim()).expand(-1,-1,-1,-1,7)

        rm_pos = reg * pos_equal_one_for_reg
        targets_pos = targets * pos_equal_one_for_reg

        cls_pos_loss = -pos_equal_one *  torch.log(p_pos + 1e-6)
        cls_pos_loss = cls_pos_loss.sum() / (pos_equal_one.sum() + 1e-6)

        cls_neg_loss = -neg_equal_one *  torch.log(1 - p_pos + 1e-6)
        cls_neg_loss = cls_neg_loss.sum() / (neg_equal_one.sum() + 1e-6)

        reg_loss = self.smoothl1loss(rm_pos, targets_pos)
        reg_loss = reg_loss / (pos_equal_one.sum() + 1e-6)
        conf_loss = self.alpha * cls_pos_loss + self.beta * cls_neg_loss

        if tag == 'val':
            xyz_loss = self.smoothl1loss(rm_pos[..., [0,1,2]], targets_pos[..., [0,1,2]]) / (pos_equal_one.sum() + 1e-6)
            whl_loss = self.smoothl1loss(rm_pos[..., [3,4,5]], targets_pos[..., [3,4,5]]) / (pos_equal_one.sum() + 1e-6)
            r_loss = self.smoothl1loss(rm_pos[..., [6]], targets_pos[..., [6]]) / (pos_equal_one.sum() + 1e-6)
            return conf_loss, reg_loss, xyz_loss, whl_loss, r_loss

        return conf_loss, reg_loss, None, None, None












