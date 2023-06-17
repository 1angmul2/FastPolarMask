from audioop import avg
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.losses import sigmoid_focal_loss as _sigmoid_focal_loss
from mmdet.models.builder import LOSSES
from mmdet.models.losses.utils import weight_reduce_loss


@LOSSES.register_module()
class MaskIOULoss(nn.Module):
    def __init__(self, reduction='mean', loss_weight=1.0):
        super(MaskIOULoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight,
                reduction_override='mean', avg_factor=None):
        '''
         :param pred:  shape (N,36), N is nr_box
         :param target: shape (N,36)
         :return: loss
         '''

        total = torch.stack([pred,target], -1)
        l_max = total.max(dim=2)[0].clamp(min=1e-6)
        l_min = total.min(dim=2)[0].clamp(min=1e-6)

        loss = (l_max.sum(dim=1) / l_min.sum(dim=1)).log()
        loss = loss * weight
        if reduction_override == 'none':
            pass
        elif reduction_override == 'mean':
            loss = loss.sum()
            if avg_factor is not None:
                loss = loss / avg_factor
            else:
                loss = loss.mean()
        elif reduction_override == 'sum':
            loss = loss.sum()
        else:
            raise NotImplementedError
        return loss * self.loss_weight


@LOSSES.register_module
class MaskIOULoss_v2(nn.Module):
    def __init__(self):
        super(MaskIOULoss_v2, self).__init__()

    def forward(self, pred, target, weight, avg_factor=None):
        '''
         :param pred:  shape (N,36), N is nr_box
         :param target: shape (N,36)
         :return: loss
         '''

        total = torch.stack([pred,target], -1)
        l_max = total.max(dim=2)[0]
        l_min = total.min(dim=2)[0].clamp(min=1e-6)

        # loss = (l_max.sum(dim=1) / l_min.sum(dim=1)).log()
        loss = (l_max / l_min).log().mean(dim=1)
        loss = loss * weight
        loss = loss.sum() / avg_factor
        return loss


@LOSSES.register_module
class MaskIOULoss_v3(nn.Module):
    def __init__(self):
        super(MaskIOULoss_v3, self).__init__()

    def forward(self, pred, target, weight, avg_factor=None):
        '''
         :param pred:  shape (N,36), N is nr_box
         :param target: shape (N,36)
         :return: loss
         '''

        total = torch.stack([pred,target], -1)
        l_max = total.max(dim=2)[0].pow(2)
        l_min = total.min(dim=2)[0].pow(2)

        # loss = 2 * (l_max.prod(dim=1) / l_min.prod(dim=1)).log()
        # loss = 2 * (l_max.log().sum(dim=1) - l_min.log().sum(dim=1))
        loss = (l_max.sum(dim=1) / l_min.sum(dim=1)).log()
        loss = loss * weight
        loss = loss.sum() / avg_factor
        return loss

