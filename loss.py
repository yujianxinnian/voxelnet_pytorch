import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
#（1）生成200x176x2=70400个anchor，每个anchor有0和90度朝向，所以乘以两倍。后续特征图大小为（200x176），相当于每个特征生成两个anchor。anchor的属性包括x、y、z、h、w、l、rz，即70400x7。
# 2）通过计算anchor和目标框在xoy平面内外接矩形的iou来判断anchor是正样本还是负样本。正样本的iou 阈值为0.6，负样本iou阈值为0.45。正样本还必须包括iou最大的anchor，负样本必须不包含iou最大的anchor。
#（3）由于anchors的维度表示为200x176x2，用维度为200x176x2矩阵pos_equal_one来表示正样本anchor，取值为1的位置表示anchor为正样本，否则为0。
#（4）同样地，用维度为200x176x2矩阵neg_equal_one来表示负样本anchor，取值为1的位置表示anchor为负样本，否则为0
#（5）用targets来表示anchor与真实检测框之间的差异，包含x、y、z、h、w、l、rz等7个属性之间的差值，这跟后续损失函数直接相关。targets维度为200x176x14，最后一个维度的前7维表示rz=0的情况，后7维表示rz=pi/2的情况。

class VoxelLoss(nn.Module):
    def __init__(self, alpha, beta, gamma):
        super(VoxelLoss, self).__init__()
        self.smoothl1loss = nn.SmoothL1Loss(reduction='sum',beta=gamma)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    def forward(self, reg, p_pos, pos_equal_one, neg_equal_one, targets, tag='train'):
        prob_output = p_pos
        delta_output = reg
        pos_equal_one_for_reg = np.concatenate(
            [np.tile(pos_equal_one[..., [0]], 7), np.tile(pos_equal_one[..., [1]], 7)], axis = -1)
        pos_equal_one_sum = np.clip(np.sum(pos_equal_one, axis = (1, 2, 3)).reshape(-1, 1, 1, 1), a_min = 1, a_max = None)
        neg_equal_one_sum = np.clip(np.sum(neg_equal_one, axis = (1, 2, 3)).reshape(-1, 1, 1, 1), a_min = 1, a_max = None)

        # Move to gpu
        device = reg.device
        pos_equal_one = torch.from_numpy(pos_equal_one).to(device).float()
        neg_equal_one = torch.from_numpy(neg_equal_one).to(device).float()
        targets = torch.from_numpy(targets).to(device).float()
        pos_equal_one_for_reg = torch.from_numpy(pos_equal_one_for_reg).to(device).float()
        pos_equal_one_sum = torch.from_numpy(pos_equal_one_sum).to(device).float()
        neg_equal_one_sum = torch.from_numpy(neg_equal_one_sum).to(device).float()
        # [batch, cfg.FEATURE_HEIGHT, cfg.FEATURE_WIDTH, 2/14] -> [batch, 2/14, cfg.FEATURE_HEIGHT, cfg.FEATURE_WIDTH]
        pos_equal_one = pos_equal_one.permute(0, 3, 1, 2)
        neg_equal_one = neg_equal_one.permute(0, 3, 1, 2)
        targets = targets.permute(0, 3, 1, 2)
        pos_equal_one_for_reg = pos_equal_one_for_reg.permute(0, 3, 1, 2)
        # Calculate loss
        rm_pos = delta_output * pos_equal_one_for_reg
        targets_pos = targets * pos_equal_one_for_reg
        reg_loss = smooth_l1(rm_pos, targets_pos, self.gamma) / pos_equal_one_sum
        reg_loss = torch.sum(reg_loss)

        cls_pos_loss = (-pos_equal_one * torch.log(prob_output + 1e-6)) / pos_equal_one_sum
        cls_neg_loss = (-neg_equal_one * torch.log(1 - prob_output + 1e-6)) / neg_equal_one_sum

        cls_loss = torch.sum(self.alpha * cls_pos_loss + self.beta * cls_neg_loss)
        cls_pos_loss_rec = torch.sum(cls_pos_loss)
        cls_neg_loss_rec = torch.sum(cls_neg_loss)

        
        loss = cls_loss + reg_loss

        return loss, cls_loss, reg_loss, cls_pos_loss_rec, cls_neg_loss_rec

def smooth_l1(deltas, targets, sigma = 3.0):
    # Reference: https://mohitjainweb.files.wordpress.com/2018/03/smoothl1loss.pdf
    sigma2 = sigma * sigma
    diffs = deltas - targets
    smooth_l1_signs = torch.lt(torch.abs(diffs), 1.0 / sigma2).float()

    smooth_l1_option1 = torch.mul(diffs, diffs) * 0.5 * sigma2
    smooth_l1_option2 = torch.abs(diffs) - 0.5 / sigma2
    smooth_l1_add = torch.mul(smooth_l1_option1, smooth_l1_signs) + torch.mul(smooth_l1_option2, 1 - smooth_l1_signs)
    smooth_l1 = smooth_l1_add

    return smooth_l1       


    # def forward(self, reg, p_pos, pos_equal_one, neg_equal_one, targets, tag='train'):
    #     # reg (N * A*7 * H * W) , score (N * A * H * W)
    #     # pos_equal_one, neg_equal_one(B,H,W,A)，这里存放的正样本和负样本的标签，是就是1，不是这个位置就是0
    #     # A表示每个位置放置的anchor数，这里是2一个0度一个90度
        
    #     reg = reg.permute(0,2,3,1).contiguous()
    #     reg = reg.view(reg.size(0),reg.size(1),reg.size(2),-1,7) # (N * H * W * A * 7)
    #     targets = targets.view(targets.size(0),targets.size(1),targets.size(2),-1,7) # (N * H * W * A * 7)
    #     pos_equal_one_for_reg = pos_equal_one.unsqueeze(pos_equal_one.dim()).expand(-1,-1,-1,-1,7) #(B,H,W,A,7)
    #     rm_pos = reg * pos_equal_one_for_reg
    #     targets_pos = targets * pos_equal_one_for_reg
    #      #只计算正样本的回归损失
    #     reg_loss = self.smoothl1loss(rm_pos, targets_pos)
    #     reg_loss = reg_loss / (pos_equal_one.sum() + 1e-6) # 1e-6 可以保证除数不为零

    #     p_pos = F.sigmoid(p_pos.permute(0,2,3,1))  # skyhehe123 中的代码，原本是在VoxelNet中做的
    #     #这里是正样本的分类损失
    #     cls_pos_loss = -pos_equal_one *  torch.log(p_pos + 1e-6)
    #     cls_pos_loss = cls_pos_loss.sum() / (pos_equal_one.sum() + 1e-6)
    #     #这里是负样本的分类损失
    #     cls_neg_loss = -neg_equal_one *  torch.log(1 - p_pos + 1e-6)
    #     cls_neg_loss = cls_neg_loss.sum() / (neg_equal_one.sum() + 1e-6)
    #     # 总损失
    #     conf_loss = self.alpha * cls_pos_loss + self.beta * cls_neg_loss

    #     if tag == 'val':
    #         xyz_loss = self.smoothl1loss(rm_pos[..., [0,1,2]], targets_pos[..., [0,1,2]]) / (pos_equal_one.sum() + 1e-6)
    #         whl_loss = self.smoothl1loss(rm_pos[..., [3,4,5]], targets_pos[..., [3,4,5]]) / (pos_equal_one.sum() + 1e-6)
    #         r_loss = self.smoothl1loss(rm_pos[..., [6]], targets_pos[..., [6]]) / (pos_equal_one.sum() + 1e-6)
    #         return conf_loss, reg_loss, xyz_loss, whl_loss, r_loss

    #     return conf_loss, reg_loss, None, None, None












