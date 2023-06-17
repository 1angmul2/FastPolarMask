# Copyright (c) OpenMMLab. All rights reserved.
import random

import torch
import torch.distributed as dist
import torch.nn.functional as F
from mmcv.runner import get_dist_info

from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.single_stage import SingleStageDetector


@DETECTORS.register_module()
class PicoDet(SingleStageDetector):
    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 sybn=False,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 input_size=(640, 640),
                 size_multiplier=32,
                 random_size_range=(15, 25),
                 random_size_interval=10,
                 init_cfg=None):
        super(PicoDet, self).__init__(backbone, neck, bbox_head, train_cfg,
                                    test_cfg, pretrained, init_cfg)
        self.rank, self.world_size = get_dist_info()
        self._default_input_size = input_size
        self._input_size = input_size
        self._random_size_range = random_size_range
        self._random_size_interval = random_size_interval
        self._size_multiplier = size_multiplier
        self._progress_in_iter = 0

        if sybn:
            torch.nn.SyncBatchNorm.convert_sync_batchnorm(self)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        # Multi-scale training
        img, gt_bboxes = self._preprocess(img, gt_bboxes)

        losses = super(PicoDet, self).forward_train(img, img_metas, gt_bboxes,
                                                  gt_labels, gt_bboxes_ignore)

        # random resizing
        if (self._progress_in_iter + 1) % self._random_size_interval == 0:
            self._input_size = self._random_resize()
        self._progress_in_iter += 1

        return losses

    def _preprocess(self, img, gt_bboxes):
        scale_y = self._input_size[0] / self._default_input_size[0]
        scale_x = self._input_size[1] / self._default_input_size[1]
        if scale_x != 1 or scale_y != 1:
            mode = self.get_mode()
            align_corners = False \
                if mode in ('bilinear', 'bicubic', 'trilinear') \
                    else None
            img = F.interpolate(
                img,
                size=self._input_size,
                mode=mode,
                align_corners=align_corners
                )
            for gt_bbox in gt_bboxes:
                gt_bbox[..., 0::2] = gt_bbox[..., 0::2] * scale_x
                gt_bbox[..., 1::2] = gt_bbox[..., 1::2] * scale_y
        return img, gt_bboxes

    def _random_resize(self):
        tensor = torch.LongTensor(2).cuda()

        if self.rank == 0:
            size = random.randint(*self._random_size_range)
            size = (self._size_multiplier * size, self._size_multiplier * size)
            tensor[0] = size[0]
            tensor[1] = size[1]

        if self.world_size > 1:
            dist.barrier()
            dist.broadcast(tensor, 0)

        input_size = (tensor[0].item(), tensor[1].item())
        return input_size

    @staticmethod
    def get_mode():
        return random.choice(
            ['nearest', 'bilinear',
             'bicubic', 'area'])
