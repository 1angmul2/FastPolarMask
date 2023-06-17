# Copyright (c) OpenMMLab. All rights reserved.
import random

import torch
import torch.distributed as dist
import torch.nn.functional as F
from mmcv.runner import get_dist_info

from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.single_stage_instance_seg import SingleStageInstanceSegmentor


@DETECTORS.register_module()
class PolarMask(SingleStageInstanceSegmentor):
    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 mask_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 sybn=False,
                 input_size=(640, 640),
                 size_multiplier=32,
                 random_size_range=(15, 25),
                 random_size_interval=10,
                 init_cfg=None,
                 pretrained=None):
        self.rank, self.world_size = get_dist_info()
        self._default_input_size = input_size
        self._input_size = input_size
        self._random_size_range = random_size_range
        self._random_size_interval = random_size_interval
        self._size_multiplier = size_multiplier
        self._progress_in_iter = 0
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            mask_head=mask_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            pretrained=pretrained)
        # 在建完模型的时候直接转
        # 不然配置很麻烦
        if sybn:
            torch.nn.SyncBatchNorm.convert_sync_batchnorm(self)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_masks,
                      gt_labels,
                      gt_bboxes=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        # Multi-scale training
        (img, gt_bboxes, kwargs['mask_centers'],
         kwargs['mask_contours']) = self._preprocess(
            img, gt_bboxes, kwargs['mask_centers'],
            kwargs['mask_contours'])
        
        x = self.extract_feat(img)
        losses = dict()

        # CondInst and YOLACT have bbox_head
        if self.bbox_head:
            # bbox_head_preds is a tuple
            bbox_head_preds = self.bbox_head(x)
            # positive_infos is a list of obj:`InstanceData`
            # It contains the information about the positive samples
            # CondInst, YOLACT
            det_losses, positive_infos = self.bbox_head.loss(
                *bbox_head_preds,
                gt_bboxes=gt_bboxes,
                gt_labels=gt_labels,
                gt_masks=gt_masks,
                img_metas=img_metas,
                gt_bboxes_ignore=gt_bboxes_ignore,
                **kwargs)
            losses.update(det_losses)
        else:
            positive_infos = None

        mask_loss = self.mask_head.forward_train(
            x,
            gt_labels,
            gt_masks,
            img_metas,
            positive_infos=positive_infos,
            gt_bboxes=gt_bboxes,
            gt_bboxes_ignore=gt_bboxes_ignore,
            **kwargs)
        # avoid loss override
        assert not set(mask_loss.keys()) & set(losses.keys())

        losses.update(mask_loss)
        
        # random resizing
        if (self._progress_in_iter + 1) % self._random_size_interval == 0:
            self._input_size = self._random_resize()
        self._progress_in_iter += 1
        
        return losses

    def set_epoch(self, epoch):
        self.mask_head.epoch = epoch

    def _preprocess(self, img, gt_bboxes, mask_centers, mask_contours):
        scale_y = self._input_size[0] / self._default_input_size[0]
        scale_x = self._input_size[1] / self._default_input_size[1]
        if scale_x != 1 or scale_y != 1:
            img = F.interpolate(
                img,
                size=self._input_size,
                # mode='bilinear',
                mode=self.get_mode(),
                # align_corners=False
                )
            for gt_bbox in gt_bboxes:
                gt_bbox[..., 0::2] = gt_bbox[..., 0::2] * scale_x
                gt_bbox[..., 1::2] = gt_bbox[..., 1::2] * scale_y
            for mask_center in mask_centers:
                mask_center[..., 0::2] = mask_center[..., 0::2] * scale_x
                mask_center[..., 1::2] = mask_center[..., 1::2] * scale_y
            for mask_contour in mask_contours:
                mask_contour[..., 0::2] = mask_contour[..., 0::2] * scale_x
                mask_contour[..., 1::2] = mask_contour[..., 1::2] * scale_y
        return img, gt_bboxes, mask_centers, mask_contours

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

    def forward_dummy(self, img, *args, **kwargs):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        # get origin input shape to support onnx dynamic shape
        cls_scores, pts_offsets = outs
        cls_score0, cls_score1, cls_score2 = cls_scores
        pts_offset0, pts_offset1, pts_offset2 = pts_offsets
        outs = (cls_score0, pts_offset0,
                cls_score1, pts_offset1,
                cls_score2, pts_offset2)
        if kwargs.get('forward_only', True):
            return outs
        
        raise NotImplementedError
