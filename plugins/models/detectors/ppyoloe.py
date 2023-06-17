# Copyright (c) OpenMMLab. All rights reserved.
import random

import torch
import torch.distributed as dist
import torch.nn.functional as F
from mmcv.runner import get_dist_info

from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.single_stage import SingleStageDetector
from .picodet import PicoDet


@DETECTORS.register_module()
class PPYOLOE(PicoDet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_epoch(self, epoch):
        self.bbox_head.epoch = epoch

    def init_weights(self):
        return super().init_weights()

