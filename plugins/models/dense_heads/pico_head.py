# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
from torch._C import dtype
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, Scale, DepthwiseSeparableConvModule, bias_init_with_prob
from mmcv.runner import force_fp32

from mmdet.core import (MlvlPointGenerator, anchor_inside_flags, bbox_overlaps, build_assigner,
                        build_sampler, images_to_levels, multi_apply,
                        reduce_mean, unmap)
from mmdet.core.bbox.builder import build_bbox_coder
from mmdet.core.utils import filter_scores_and_topk
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.dense_heads.anchor_head import AnchorHead
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead
from mmdet.models.dense_heads.dense_test_mixins import BBoxTestMixin
from mmdet.models.dense_heads.gfl_head import Integral, GFLHead
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead


@HEADS.register_module()
class PicoHead(BaseDenseHead, BBoxTestMixin):
    """

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of hidden channels in stacking convs.
            Default: 256
        stacked_convs (int): Number of stacking convs of the head.
            Default: 2.
        strides (tuple): Downsample factor of each feature map.
        use_depthwise (bool): Whether to depthwise separable convolution in
            blocks. Default: False
        dcn_on_last_conv (bool): If true, use dcn in the last layer of
            towers. Default: False.
        conv_bias (bool | str): If specified as `auto`, it will be decided by
            the norm_cfg. Bias of conv will be set as True if `norm_cfg` is
            None, otherwise False. Default: "auto".
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer. Default: None.
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        loss_obj (dict): Config of objectness loss.
        loss_l1 (dict): Config of L1 loss.
        train_cfg (dict): Training config of anchor head.
        test_cfg (dict): Testing config of anchor head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=128,
                 stacked_convs=4,
                 strides=[8, 16, 32, 64],
                 share_cls_reg=True,
                 reg_max=7,
                 cell_offset=0.5,
                 use_depthwise=True,
                 dcn_on_last_conv=False,
                 conv_bias='auto',
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='HSwish'),
                 bbox_coder=dict(type='DistancePointBBoxCoder'),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     reduction='sum',
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='IoULoss',
                     mode='square',
                     eps=1e-16,
                     reduction='sum',
                     loss_weight=5.0),
                 loss_dfl=dict(
                     type='DistributionFocalLoss',
                     loss_weight=0.25),
                 loss_l1=dict(type='L1Loss', reduction='sum', loss_weight=1.0),
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=dict(
                     type='Kaiming',
                     layer='Conv2d',
                     a=math.sqrt(5),
                     distribution='uniform',
                     mode='fan_in',
                     nonlinearity='leaky_relu')
                 ):

        super().__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.cls_out_channels = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.share_cls_reg = share_cls_reg
        self.reg_max = reg_max
        self.use_depthwise = use_depthwise
        self.dcn_on_last_conv = dcn_on_last_conv
        assert conv_bias == 'auto' or isinstance(conv_bias, bool)
        self.conv_bias = conv_bias
        self.use_sigmoid_cls = True

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_dfl = build_loss(loss_dfl)

        self.use_l1 = False  # This flag will be modified by hooks.
        self.loss_l1 = build_loss(loss_l1)

        self.bbox_coder = build_bbox_coder(bbox_coder)

        self.prior_generator = MlvlPointGenerator(strides, offset=cell_offset)

        self.test_cfg = test_cfg
        self.train_cfg = train_cfg

        self.sampling = False
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            # sampling=False so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)

        self.fp16_enabled = False
        self.integral = Integral(self.reg_max)
        self._init_layers()

    def _init_layers(self):
        self.multi_level_cls_convs = nn.ModuleList()
        if not self.share_cls_reg:
            self.multi_level_reg_convs = nn.ModuleList()
        else:
            self.multi_level_reg_convs = []
        self.multi_level_conv_cls = nn.ModuleList()
        self.multi_level_conv_reg = nn.ModuleList()
        for _ in self.strides:
            self.multi_level_cls_convs.append(self._build_stacked_convs())
            if not self.share_cls_reg:
                self.multi_level_reg_convs.append(self._build_stacked_convs())
            else:
                self.multi_level_reg_convs.append([])
            conv_cls, conv_reg = self._build_predictor()
            self.multi_level_conv_cls.append(conv_cls)
            self.multi_level_conv_reg.append(conv_reg)

    def _build_stacked_convs(self):
        """Initialize conv layers of a single level head."""
        conv = DepthwiseSeparableConvModule \
            if self.use_depthwise else ConvModule
        stacked_convs = []
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            if self.dcn_on_last_conv and i == self.stacked_convs - 1:
                conv_cfg = dict(type='DCNv2')
            else:
                conv_cfg = self.conv_cfg
            stacked_convs.append(
                conv(
                    chn,
                    self.feat_channels,
                    5,
                    stride=1,
                    padding=2,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    bias=self.conv_bias))
        return nn.Sequential(*stacked_convs)

    def _build_predictor(self):
        """Initialize predictor layers of a single level head."""
        cls_out_channels = self.cls_out_channels + 4 * (self.reg_max + 1)\
                    if self.share_cls_reg else self.cls_out_channels
        conv_cls = nn.Conv2d(self.feat_channels, cls_out_channels, 1)
        conv_reg = nn.Conv2d(self.feat_channels, self.reg_max+1, 1)
        return conv_cls, conv_reg

    def init_weights(self):
        super(PicoHead, self).init_weights()
        # Use prior in model initialization to improve stability
        bias_init = bias_init_with_prob(0.01)
        for conv_cls in self.multi_level_conv_cls:
            conv_cls.weight.data.fill_(0.0)
            conv_cls.bias.data.fill_(bias_init)
        for conv_reg in self.multi_level_conv_reg:
            conv_reg.weight.data.fill_(0.0)
            conv_reg.bias.data.fill_(1.0)

    def forward_single(self, x, cls_convs, reg_convs, conv_cls, conv_reg):
        """Forward feature of a single scale level."""

        cls_feat = x
        cls_feat = cls_convs(cls_feat)
        if not self.share_cls_reg:
            reg_feat = reg_convs(x)
            cls_score = conv_cls(cls_feat)
            bbox_pred = conv_reg(reg_feat)
        else:
            cls_logits = conv_cls(cls_feat)
            cls_score, bbox_pred = torch.split(
                cls_logits,
                [self.cls_out_channels, 4 * (self.reg_max + 1)],
                dim=1)

        return cls_score, bbox_pred

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            tuple[Tensor]: A tuple of multi-level predication map, each is a
                4D-tensor of shape (batch_size, 5+num_classes, height, width).
        """

        return multi_apply(self.forward_single, feats,
                           self.multi_level_cls_convs,
                           self.multi_level_reg_convs,
                           self.multi_level_conv_cls,
                           self.multi_level_conv_reg)

    def _bboxes_nms(self, cls_scores, bboxes, score_factor, cfg):
        max_scores, labels = torch.max(cls_scores, 1)
        valid_mask = score_factor * max_scores >= cfg.score_thr

        bboxes = bboxes[valid_mask]
        scores = max_scores[valid_mask] * score_factor[valid_mask]
        labels = labels[valid_mask]

        if labels.numel() == 0:
            return bboxes, labels
        else:
            dets, keep = batched_nms(bboxes, scores, labels, cfg.nms)
            return dets, labels[keep]

    def _loss_single(self, cls_score, bbox_pred, anchors, labels, label_weights,
                     bbox_targets, stride, num_total_samples):
        """Compute loss of a single scale level.

        Args:
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            cls_score (Tensor): Cls and quality joint scores for each scale
                level has shape (N, num_classes, H, W).
            bbox_pred (Tensor): Box distribution logits for each scale
                level with shape (N, 4*(n+1), H, W), n is max value of integral
                set.
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor
                weight shape (N, num_total_anchors, 4).
            stride (tuple): Stride in this scale level.
            num_total_samples (int): Number of positive samples that is
                reduced over all GPUs.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert stride[0] == stride[1], 'h stride is not equal to w stride!'
        anchors = anchors[None].expand(
            (cls_score.shape[0],) + anchors.shape).reshape(-1, 4)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        bbox_pred = bbox_pred.permute(0, 2, 3,
                                      1).reshape(-1, 4 * (self.reg_max + 1))
        bbox_targets = bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero(as_tuple=False).squeeze(1)
        score = label_weights.new_zeros(cls_score.shape)

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_anchors = anchors[pos_inds]
            pos_anchor_centers = pos_anchors[:, :2] / stride[0]

            weight_targets = cls_score.detach().sigmoid()
            weight_targets = weight_targets.max(dim=1)[0][pos_inds]
            pos_bbox_pred_corners = self.integral(pos_bbox_pred)
            pos_decode_bbox_pred = self.bbox_coder.decode(
                pos_anchor_centers, pos_bbox_pred_corners)
            pos_decode_bbox_targets = pos_bbox_targets / stride[0]
            score[pos_inds, labels[pos_inds]] = bbox_overlaps(
                pos_decode_bbox_pred.detach(),
                pos_decode_bbox_targets,
                is_aligned=True)
            pred_corners = pos_bbox_pred.reshape(-1, self.reg_max + 1)
            target_corners = self.bbox_coder.encode(pos_anchor_centers,
                                                    pos_decode_bbox_targets,
                                                    self.reg_max).reshape(-1)

            # regression loss
            loss_bbox = self.loss_bbox(
                pos_decode_bbox_pred,
                pos_decode_bbox_targets,
                weight=weight_targets,
                avg_factor=1.0)

            # dfl loss
            loss_dfl = self.loss_dfl(
                pred_corners,
                target_corners,
                weight=weight_targets[:, None].expand(-1, 4).reshape(-1),
                avg_factor=4.0)
        else:
            loss_bbox = bbox_pred.sum() * 0
            loss_dfl = bbox_pred.sum() * 0
            weight_targets = bbox_pred.new_tensor(0)

        # cls (vfl) loss
        loss_cls = self.loss_cls(
            cls_score, score,
            weight=None,
            avg_factor=num_total_samples)

        return loss_cls, loss_bbox, loss_dfl, weight_targets.sum()

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Cls and quality scores for each scale
                level has shape (N, num_classes, H, W).
            bbox_preds (list[Tensor]): Box distribution logits for each scale
                level with shape (N, 4*(n+1), H, W), n is max value of integral
                set.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (list[Tensor] | None): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        device = cls_scores[0].device

        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=device,
            with_stride=True)

        cls_reg_targets = self.get_targets(
            cls_scores,
            bbox_preds,
            gt_bboxes,
            gt_labels,
            list(map(lambda x: x[0]*x[1], featmap_sizes)),
            mlvl_priors,
            img_metas=img_metas)
        if cls_reg_targets is None:
            return None
        
        (flatten_cls_preds, flatten_bbox_deintegral, flatten_bbox_integral, 
         labels_all, bbox_targets_all, pos_anchor_all,
         num_total_pos, num_total_neg) = cls_reg_targets
        
        num_total_samples = reduce_mean(
            torch.tensor(num_total_pos, dtype=torch.float,
                        device=device)).item()
        num_total_samples = max(num_total_samples, 1.0)
        
        if num_total_pos >= 1:
            pos_anchor_centers = pos_anchor_all[:, :2] / pos_anchor_all[:, 2:]
            
            pos_inds = ((labels_all >= 0)
                        & (labels_all < self.num_classes)).nonzero(as_tuple=False).squeeze(1)
            score = torch.zeros_like(flatten_cls_preds)

            weight_targets = flatten_cls_preds.detach()[pos_inds].max(
                dim=1)[0].sigmoid()
            pos_bbox_integral = flatten_bbox_integral[pos_inds]
            pos_decode_bbox_pred = self.bbox_coder.decode(
                pos_anchor_centers, pos_bbox_integral)

            pos_decode_bbox_targets = bbox_targets_all / pos_anchor_all[:, 2:].repeat(1, 2)
            
            score[pos_inds, labels_all[pos_inds]] = bbox_overlaps(
                pos_decode_bbox_pred.detach(),
                pos_decode_bbox_targets,
                is_aligned=True)
            
            pred_corners = flatten_bbox_deintegral[pos_inds].reshape(-1, self.reg_max + 1)
            target_corners = self.bbox_coder.encode(pos_anchor_centers,
                                                    pos_decode_bbox_targets,
                                                    self.reg_max).reshape(-1)

            # regression loss
            losses_bbox = self.loss_bbox(
                pos_decode_bbox_pred,
                pos_decode_bbox_targets,
                weight=weight_targets,
                avg_factor=1.0)

            # dfl loss
            losses_dfl = self.loss_dfl(
                pred_corners,
                target_corners,
                weight=weight_targets[:, None].expand(-1, 4).reshape(-1),
                avg_factor=4.0)

            # cls (vfl) loss
            losses_cls = self.loss_cls(
                flatten_cls_preds, score,
                weight=None,
                avg_factor=num_total_samples)

            avg_factor = weight_targets.sum()
            avg_factor = reduce_mean(avg_factor).clamp_(min=1).item()
            losses_bbox = losses_bbox / avg_factor
            losses_dfl = losses_dfl / avg_factor
        else:
            losses_bbox = sum(cls_scores.sum())*0.
            losses_dfl = sum(bbox_preds.sum())*0.

        return dict(
            loss_cls=losses_cls, loss_bbox=losses_bbox, loss_dfl=losses_dfl)

    def _get_targets(self, cls_scores, bbox_preds, gt_bboxes_list,
                    gt_labels_list, num_points, mlvl_priors, img_metas):
        num_levels = len(self.strides)
        num_imgs = len(img_metas)
        img_lvl_cls_preds = [
            cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                 self.cls_out_channels)
            for cls_pred in cls_scores
        ]
        img_lvl_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4*(self.reg_max+1))
            for bbox_pred in bbox_preds
        ]
        
        img_lvl_cls_preds = torch.cat(img_lvl_cls_preds, dim=1)
        img_lvl_bbox_preds = torch.cat(img_lvl_bbox_preds, dim=1)
        img_lvl_bbox_integral = self.integral(
            img_lvl_bbox_preds).view(num_imgs, -1, 4)
        concat_priors = torch.cat(mlvl_priors, dim=0)
        
        # [lvl*h*w, 2]
        flatten_centers_w_stride = \
            (concat_priors[:, :2] / concat_priors[:, 2:])[
                None].expand(num_imgs, concat_priors.shape[0], 2)
        # apply strides
        img_lvl_bbox_preds = self.bbox_coder.decode(
            flatten_centers_w_stride,
            img_lvl_bbox_integral
            ) * concat_priors[None, :, 2:].repeat(1, 1, 2)

        # get labels and bbox_targets of each image
        (all_labels, all_label_weights, all_bbox_targets,
         all_bbox_weights, pos_inds_list,
         neg_inds_list) = multi_apply(
            self._get_target_single,
            img_lvl_cls_preds,
            img_lvl_bbox_preds,
            gt_bboxes_list,
            gt_labels_list,
            img_metas,
            num_points=num_points,
            priors=concat_priors)

        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        labels_list = images_to_levels(all_labels, num_points)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_points)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_points)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_points)
        return (labels_list, label_weights_list,
                bbox_targets_list, bbox_weights_list, num_total_pos,
                num_total_neg)

    def get_targets(self, cls_scores, bbox_preds, gt_bboxes_list,
                    gt_labels_list, num_points, mlvl_priors, img_metas):
        num_imgs = len(img_metas)
        img_lvl_cls_preds = torch.cat([
            cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                 self.cls_out_channels)
            for cls_pred in cls_scores
        ], dim=1)
        img_lvl_bbox_deintegral = torch.cat([
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4*(self.reg_max+1))
            for bbox_pred in bbox_preds
        ], dim=1)
        
        # img_lvl_cls_preds = torch.cat(img_lvl_cls_preds, dim=1)
        # img_lvl_bbox_deintegral = torch.cat(img_lvl_bbox_deintegral, dim=1)
        img_lvl_bbox_integral = self.integral(
            img_lvl_bbox_deintegral).view(num_imgs, -1, 4)
        concat_priors = torch.cat(mlvl_priors, dim=0)
        
        # [lvl*h*w, 2]
        flatten_centers_w_stride = \
            (concat_priors[:, :2] / concat_priors[:, 2:])[
                None].expand(num_imgs, concat_priors.shape[0], 2)

        # apply strides
        img_lvl_bbox_decode = self.bbox_coder.decode(
            flatten_centers_w_stride,
            img_lvl_bbox_integral.detach()
            ) * concat_priors[None, :, 2:].repeat(1, 1, 2)

        # get labels and bbox_targets of each image
        (all_labels, all_bbox_targets, all_pos_anchors,
         pos_inds_list, neg_inds_list) = multi_apply(
            self._get_target_single,
            img_lvl_cls_preds.detach(),
            img_lvl_bbox_decode.detach(),
            gt_bboxes_list,
            gt_labels_list,
            img_metas,
            num_points=num_points,
            priors=concat_priors)

        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        
        chn = img_lvl_cls_preds.shape[-1]
        # flatten
        img_lvl_cls_preds = img_lvl_cls_preds.reshape(-1, chn)
        img_lvl_bbox_integral = img_lvl_bbox_integral.reshape(-1, 4)
        img_lvl_bbox_deintegral = img_lvl_bbox_deintegral.reshape(
            -1, 4*(self.reg_max+1))
        all_labels = torch.cat(all_labels, dim=0)
        all_bbox_targets = torch.cat(all_bbox_targets, dim=0)
        all_pos_anchors = torch.cat(all_pos_anchors, dim=0)

        return (img_lvl_cls_preds, img_lvl_bbox_deintegral, img_lvl_bbox_integral,
                all_labels, all_bbox_targets, all_pos_anchors,
                num_total_pos, num_total_neg)

    @torch.no_grad()
    def _get_target_single(self,
                           cls_preds,
                           decoded_bboxes,
                           gt_bboxes,
                           gt_labels,
                           img_meta,
                           num_points,
                           priors):
        """Compute classification and regression targets for
        priors in a single image.
        Args:
            cls_preds (Tensor): Classification predictions of one image,
                a 2D-Tensor with shape [num_points, num_classes]
            priors (Tensor): All priors of one image, a 2D-Tensor with shape
                [num_points, 4] in [cx, xy, stride_w, stride_y] format.
            decoded_bboxes (Tensor): Decoded bboxes predictions of one image,
                a 2D-Tensor with shape [num_points, 4] in [tl_x, tl_y,
                br_x, br_y] format.
            gt_bboxes (Tensor): Ground truth bboxes of one image, a 2D-Tensor
                with shape [num_gts, 4] in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth labels of one image, a Tensor
                with shape [num_gts].
        """

        num_points = priors.size(0)
        num_gts = gt_labels.size(0)
        gt_bboxes = gt_bboxes.to(decoded_bboxes.dtype)
        
        # No target
        if num_gts == 0:
            labels = decoded_bboxes.new_full((num_points, ),
                                            self.num_classes,
                                            dtype=torch.long)
            bbox_targets = cls_preds.new_zeros((0, 4))
            pos_anchors = cls_preds.new_zeros((0, 4))
            pos_inds = cls_preds.new_zeros((0, ), dtype=torch.int64)
            neg_inds = torch.arange(num_points,
                                    device=cls_preds.device,
                                    dtype=torch.int64)
            return labels, bbox_targets, pos_anchors, pos_inds, neg_inds

        assign_result = self.assigner.assign(
            cls_preds.sigmoid(),
            priors, decoded_bboxes, gt_bboxes, gt_labels)

        sampling_result = self.sampler.sample(assign_result, priors, gt_bboxes)
    
        labels = decoded_bboxes.new_full((num_points, ),
                                         self.num_classes,
                                         dtype=torch.long)
        label_weights = decoded_bboxes.new_ones(num_points,
                                                dtype=torch.float32)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        
        labels[pos_inds] = gt_labels[
            sampling_result.pos_assigned_gt_inds]

        bbox_targets = sampling_result.pos_gt_bboxes
        
        pos_anchors = sampling_result.pos_bboxes
        return labels, bbox_targets, pos_anchors, pos_inds, neg_inds

    @torch.no_grad()
    def __get_target_single(self,
                           cls_preds,
                           decoded_bboxes,
                           gt_bboxes,
                           gt_labels,
                           img_meta,
                           num_points,
                           priors):
        """Compute classification and regression targets for
        priors in a single image.
        Args:
            cls_preds (Tensor): Classification predictions of one image,
                a 2D-Tensor with shape [num_points, num_classes]
            priors (Tensor): All priors of one image, a 2D-Tensor with shape
                [num_points, 4] in [cx, xy, stride_w, stride_y] format.
            decoded_bboxes (Tensor): Decoded bboxes predictions of one image,
                a 2D-Tensor with shape [num_points, 4] in [tl_x, tl_y,
                br_x, br_y] format.
            gt_bboxes (Tensor): Ground truth bboxes of one image, a 2D-Tensor
                with shape [num_gts, 4] in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth labels of one image, a Tensor
                with shape [num_gts].
        """

        num_points = priors.size(0)
        num_gts = gt_labels.size(0)
        gt_bboxes = gt_bboxes.to(decoded_bboxes.dtype)
        
        # No target
        if num_gts == 0:
            cls_target = cls_preds.new_zeros((0, self.num_classes))
            bbox_target = cls_preds.new_zeros((0, 4))
            l1_target = cls_preds.new_zeros((0, 4))
            obj_target = cls_preds.new_zeros((num_points, 1))
            foreground_mask = cls_preds.new_zeros(num_points).bool()
            return (foreground_mask, cls_target, obj_target, bbox_target,
                    l1_target, 0)

        assign_result = self.assigner.assign(
            cls_preds.detach().sigmoid(),
            priors, decoded_bboxes, gt_bboxes, gt_labels)

        sampling_result = self.sampler.sample(assign_result, priors, gt_bboxes)
    
        bbox_targets = torch.zeros_like(decoded_bboxes)
        bbox_weights = torch.zeros_like(decoded_bboxes)
        labels = decoded_bboxes.new_full((num_points, ),
                                         self.num_classes,
                                         dtype=torch.long)
        label_weights = decoded_bboxes.new_zeros(num_points,
                                                 dtype=torch.float32)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]

                label_weights[pos_inds] = 1.0
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        return (labels, label_weights, bbox_targets, bbox_weights,
                pos_inds, neg_inds)
        
    def _get_bboxes_single(self,
                           cls_score_list,
                           bbox_pred_list,
                           score_factor_list,
                           mlvl_priors,
                           img_meta,
                           cfg,
                           rescale=False,
                           with_nms=True,
                           **kwargs):
        """Transform outputs of a single image into bbox predictions.

        Args:
            cls_score_list (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_points * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas from
                all scale levels of a single image, each item has shape
                (num_points * 4, H, W).
            score_factor_list (list[Tensor]): Score factor from all scale
                levels of a single image. GFL head does not need this value.
            mlvl_priors (list[Tensor]): Each element in the list is
                the priors of a single level in feature pyramid, has shape
                (num_points, 4).
            img_meta (dict): Image meta info.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            tuple[Tensor]: Results of detected bboxes and labels. If with_nms
                is False and mlvl_score_factor is None, return mlvl_bboxes and
                mlvl_scores, else return mlvl_bboxes, mlvl_scores and
                mlvl_score_factor. Usually with_nms is False is used for aug
                test. If with_nms is True, then return the following format

                - det_bboxes (Tensor): Predicted bboxes with shape \
                    [num_bboxes, 5], where the first 4 columns are bounding \
                    box positions (tl_x, tl_y, br_x, br_y) and the 5-th \
                    column are scores between 0 and 1.
                - det_labels (Tensor): Predicted labels of the corresponding \
                    box with shape [num_bboxes].
        """
        cfg = self.test_cfg if cfg is None else cfg
        img_shape = img_meta['img_shape']
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_labels = []
        for level_idx, (cls_score, bbox_pred, stride, priors) in enumerate(
                zip(cls_score_list, bbox_pred_list,
                    self.prior_generator.strides, mlvl_priors)):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            assert stride[0] == stride[1]

            bbox_pred = bbox_pred.permute(1, 2, 0)
            bbox_pred = self.integral(bbox_pred) * stride[0]

            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()

            # After https://github.com/open-mmlab/mmdetection/pull/6268/,
            # this operation keeps fewer bboxes under the same `nms_pre`.
            # There is no difference in performance for most models. If you
            # find a slight drop in performance, you can set a larger
            # `nms_pre` than before.
            results = filter_scores_and_topk(
                scores, cfg.score_thr, nms_pre,
                dict(bbox_pred=bbox_pred, priors=priors))
            scores, labels, _, filtered_results = results

            bbox_pred = filtered_results['bbox_pred']
            priors = filtered_results['priors']

            bboxes = self.bbox_coder.decode(
                priors[:, :2], bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)

        return self._bbox_post_process(
            mlvl_scores,
            mlvl_labels,
            mlvl_bboxes,
            img_meta['scale_factor'],
            cfg,
            rescale=rescale,
            with_nms=with_nms)

