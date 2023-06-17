# Copyright (c) OpenMMLab. All rights reserved.

import math
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule
from mmcv.cnn.utils.weight_init import normal_init, bias_init_with_prob, constant_init
from mmcv.cnn.bricks.activation import build_activation_layer
from mmcv.ops import batched_nms
from mmcv.runner import force_fp32

from mmdet.models.dense_heads.base_dense_head import BaseDenseHead
from mmdet.models.dense_heads.dense_test_mixins import BBoxTestMixin
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.dense_heads.gfl_head import Integral
from mmdet.core import (MlvlPointGenerator, anchor_inside_flags,
                        build_assigner, build_sampler, multi_apply,
                        filter_scores_and_topk, reduce_mean, InstanceData,
                        build_bbox_coder, select_single_mlvl)

from .ppyoloe_head import MlvlAnchorPointGenerator, ESEAttn


@HEADS.register_module()
class FastPolarMaskHead(BaseDenseHead, BBoxTestMixin):
    def __init__(self,
                 in_channels=[1024, 512, 256],
                 width_mult=1.0,
                 depth_mult=1.0,
                 num_classes=80,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='Swish'),
                 strides=[32, 16, 8],
                 grid_cell_scale=5.0,
                 grid_cell_offset=0.5,
                 reg_max=16,
                 point_num=36,
                 norm_on_bbox=False,
                 mask_coder=dict(type='DistancePointMaskCoder'),
                 bbox_coder=dict(type='DistancePointMaskCoder'),
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
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
                 loss_mask=dict(type='MaskIOULoss'),
                 use_varifocal_loss=True,
                 static_assigner=None,
                 assigner=None,
                 eval_input_size=[],
                 train_cfg=None,
                 test_cfg=None,
                 trt=False,
                 exclude_nms=False):
        super(FastPolarMaskHead, self).__init__()
        assert len(in_channels) > 0, "len(in_channels) should > 0"
        in_channels = [max(round(c * width_mult), 1) for c in in_channels]
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.cls_out_channels = num_classes
        self.strides = strides
        self.grid_cell_scale = grid_cell_scale
        self.grid_cell_offset = grid_cell_offset
        self.reg_max = reg_max
        self.point_num = point_num
        self.use_varifocal_loss = use_varifocal_loss
        self.eval_input_size = eval_input_size

        self.static_assigner = static_assigner
        self.assigner = assigner
        self.exclude_nms = exclude_nms
        
        self.act_cfg = act_cfg
        self.norm_cfg = norm_cfg
        
        self.norm_on_bbox = norm_on_bbox
        
        if self.use_varifocal_loss:
            self.loss_cls = loss_cls
        else:
            self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_mask = build_loss(loss_mask)
        self.loss_dfl = build_loss(loss_dfl)

        # self.use_l1 = False  # This flag will be modified by hooks.
        # self.loss_l1 = build_loss(loss_l1)

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.mask_coder = build_bbox_coder(mask_coder)

        self.prior_generator = MlvlAnchorPointGenerator(
            grid_cell_scale, strides, offset=grid_cell_offset)
        
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg

        self.sampling = False
        self.epoch = 0
        if self.train_cfg:
            self.initial_epoch = self.train_cfg.get('initial_epoch', 100)
            self.static_assigner = build_assigner(self.train_cfg.static_assigner)
            self.assigner = build_assigner(self.train_cfg.assigner)
            # sampling=False so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)

        self.fp16_enabled = False
        
        self.integral = Integral(self.reg_max)
        
        self._init_layers()
        self._init_weights()
    
    def _init_layers(self):
        # stem
        self.stem_cls = nn.ModuleList()
        self.stem_reg = nn.ModuleList()

        for in_c in self.in_channels:
            self.stem_cls.append(ESEAttn(in_c,
                                         norm_cfg=self.norm_cfg,
                                         act_cfg=self.act_cfg))
            self.stem_reg.append(ESEAttn(in_c, 
                                         norm_cfg=self.norm_cfg,
                                         act_cfg=self.act_cfg))
        # pred head
        self.pred_cls = nn.ModuleList()
        self.pred_reg = nn.ModuleList()
        for in_c in self.in_channels:
            self.pred_cls.append(
                nn.Conv2d(
                    in_c, self.num_classes, 3, padding=1))
            self.pred_reg.append(
                nn.Conv2d(
                    in_c, self.point_num, 3, padding=1))

    def _init_weights(self):
        prior_prob = 0.01
        for cls_, reg_ in zip(self.pred_cls, self.pred_reg):
            b = cls_.bias.view(-1, )
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            cls_.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            w = cls_.weight
            w.data.fill_(0.)
            cls_.weight = torch.nn.Parameter(w, requires_grad=True)
            
            b = reg_.bias.view(-1, )
            b.data.fill_(0.0)
            reg_.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            w = reg_.weight
            w.data.fill_(0.)
            reg_.weight = torch.nn.Parameter(w, requires_grad=True)

    def forward_train(self,
                      x,
                      gt_labels,
                      gt_masks,
                      img_metas,
                      positive_infos=None,
                      gt_bboxes=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        if positive_infos is None:
            outs = self(x)
        else:
            outs = self(x, positive_infos)

        assert isinstance(outs, tuple), 'Forward results should be a tuple, ' \
                                        'even if only one item is returned'
        loss = self.loss(
            *outs,
            gt_labels=gt_labels,
            gt_masks=gt_masks,
            img_metas=img_metas,
            gt_bboxes=gt_bboxes,
            gt_bboxes_ignore=gt_bboxes_ignore,
            positive_infos=positive_infos,
            **kwargs)
        return loss

    def forward_single(self, feat,
                       stem_cls, pred_cls,
                       stem_reg, pred_reg):
        """Forward feature of a single scale level."""
        avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
        cls_score = pred_cls(stem_cls(feat, avg_feat) + feat)
        pts_offset = pred_reg(stem_reg(feat, avg_feat))
        if self.norm_on_bbox:
            pts_offset = pts_offset.clamp(min=1e-6)
        else:
            pts_offset = pts_offset.exp()
        # cls and reg
        # [b, 80, h, w]  [b, pn, h, w]
        return cls_score, pts_offset
    
    def forward(self, feats):
        """Forward features from the upstream network.
        """
        return multi_apply(self.forward_single, feats,
                           self.stem_cls,
                           self.pred_cls,
                           self.stem_reg,
                           self.pred_reg)
    
    def get_anchors(self, featmap_sizes, img_metas, dtype, device):
        mlvl_anchors, mlvl_priors = self.prior_generator.grid_priors_anchors(
            featmap_sizes,
            dtype=dtype,
            device=device,
            with_stride=True)

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = self.prior_generator.valid_flags(
                featmap_sizes, img_meta['pad_shape'], device)
            valid_flag_list.append(multi_level_flags)
            
        return mlvl_anchors, mlvl_priors, valid_flag_list
    
    @force_fp32(apply_to=('cls_scores', 'pts_offsets'))
    def loss(self,
             cls_scores,
             pts_offsets,
             gt_labels,
             gt_masks,
             img_metas,
             gt_bboxes,
             positive_infos=None,
             gt_bboxes_ignore=None,
             mask_centers=None,
             mask_contours=None):
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        device = cls_scores[0].device
        dtype = cls_scores[0].dtype
        
        mlvl_anchors, mlvl_priors, _ = self.get_anchors(
            featmap_sizes, img_metas, dtype=dtype, device=device)
        
        # decode
        num_imgs = len(img_metas)
        # img_lvl[b, lvl*h*w, cls_n]
        il_cls_preds = torch.cat([
            cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                 self.cls_out_channels)
            for cls_pred in cls_scores
        ], dim=1)
        # [b, lvl*h*w, pn]
        il_offset_pred_w_stride = torch.cat([
            pts_offset.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.point_num)
            for pts_offset in pts_offsets
        ], dim=1)
        
        # concat all level anchors to a single tensor
        # [lvl*h*w, 2], [lvl*h*w, 4]
        concat_priors = torch.cat(mlvl_priors, dim=0)
        concat_anchors = torch.cat(mlvl_anchors, dim=0)
        
        il_offset_preds = il_offset_pred_w_stride*concat_priors[None, :, 3:]
        del il_offset_pred_w_stride

        # fake
        il_bbox_decode = il_cls_preds.new_zeros(
            (len(img_metas), 1, 4))
    
        (labels_all, bbox_targets_all, all_pos_mask_contours,
         weight_targets_all, pos_anchor_all, num_total_pos,
         num_total_neg) = self.get_targets(
            il_cls_preds.detach(),
            il_bbox_decode.detach(),
            il_offset_preds.detach(),
            gt_bboxes,
            gt_labels,
            gt_masks,
            mask_centers,
            mask_contours,
            gt_bboxes_ignore,
            [i[0]*i[1] for i in featmap_sizes],
            concat_anchors,
            concat_priors,
            img_metas=img_metas)
        if labels_all is None:
            return None
        
        il_cls_preds = il_cls_preds.reshape(-1, self.cls_out_channels)
        il_bbox_decode = il_bbox_decode.reshape(-1, 4)
        il_offset_preds = il_offset_preds.reshape(-1, self.point_num)
        
        flatten_cls_preds = il_cls_preds
        flatten_bbox_decode = il_bbox_decode
        flatten_offset_preds = il_offset_preds
        
        num_total_samples = reduce_mean(
            torch.tensor(num_total_pos, dtype=torch.float,
                        device=device)).item()
        num_total_samples = max(num_total_samples, 1.0)
        
        if num_total_pos >= 1:
            pos_anchor_centers = pos_anchor_all[:, :2]
            
            pos_inds = ((labels_all >= 0)
                        & (labels_all < self.num_classes)).nonzero(as_tuple=False).squeeze(1)

            # mask loss
            pos_offset_preds = flatten_offset_preds[pos_inds]
            # contour2dist
            gt_dist_encode = self.mask_coder.encode(
                pos_anchor_centers, all_pos_mask_contours)
            losses_mask = self.loss_mask(
                pos_offset_preds,
                gt_dist_encode,
                weight=weight_targets_all,
                reduction_override='mean',
                avg_factor=1.0)
            
            # # regression loss
            # pos_decode_bbox_targets = bbox_targets_all
            # pos_bbox_decode = flatten_bbox_decode[pos_inds]
            # losses_bbox = self.loss_bbox(
            #     pos_bbox_decode,
            #     pos_decode_bbox_targets,
            #     weight=weight_targets_all,
            #     avg_factor=1.0)
            losses_bbox = losses_mask*0.

            # cls (vfl) loss
            score = torch.zeros_like(flatten_cls_preds)
            score[pos_inds, labels_all[pos_inds]] = weight_targets_all
            if self.use_varifocal_loss:
                one_hot_label = F.one_hot(labels_all,
                                          self.num_classes + 1)[..., :-1]
                losses_cls = self._varifocal_loss(
                    flatten_cls_preds, score, one_hot_label)
            else:  # focal
                losses_cls = self.loss_cls(
                    flatten_cls_preds, labels_all,
                    weight=None,
                    avg_factor=1)

            avg_factor = weight_targets_all.sum()
            avg_factor = reduce_mean(avg_factor).clamp_(min=1).item()
            losses_cls = losses_cls / avg_factor
            losses_bbox = losses_bbox / avg_factor
            losses_mask = losses_mask / avg_factor
            
        else:
            losses_cls = flatten_cls_preds.sum()*0.
            losses_bbox = flatten_bbox_decode.sum()*0.
            losses_mask = flatten_offset_preds.sum()*0.

        return dict(
            loss_cls=losses_cls, loss_bbox=losses_bbox,
            loss_mask=losses_mask)

    def get_targets(self, il_cls_preds, il_bbox_decode, il_offset_preds,
                    gt_bboxes_list, gt_labels_list, gt_masks_list, mask_centers_list,
                    mask_contours_list, gt_bboxes_ignore_list, mlvl_num_points,
                    concat_anchors, concat_priors, img_metas):
        num_imgs = len(img_metas)

        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        # get labels and bbox_targets of each image
        (all_labels, all_bbox_targets, all_pos_anchors, all_pos_mask_contours,
         all_weight_targets, pos_inds_list, neg_inds_list) = multi_apply(
            self._get_target_single,
            il_cls_preds.detach(),
            il_bbox_decode.detach(),
            il_offset_preds.detach(),
            gt_bboxes_list,
            gt_labels_list,
            gt_masks_list,
            mask_centers_list,
            mask_contours_list,
            gt_bboxes_ignore_list,
            img_metas,
            mlvl_num_points=mlvl_num_points,
            priors=concat_priors,
            anchors=concat_anchors)

        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None, None, None, None, None, None, None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        
        all_labels = torch.cat(all_labels, dim=0)
        all_bbox_targets = torch.cat(all_bbox_targets, dim=0)
        all_weight_targets = torch.cat(all_weight_targets, dim=0)
        all_pos_anchors = torch.cat(all_pos_anchors, dim=0)
        all_pos_mask_contours = torch.cat(all_pos_mask_contours, dim=0)

        return (all_labels, all_bbox_targets, all_pos_mask_contours,
                all_weight_targets, all_pos_anchors, num_total_pos,
                num_total_neg)

    @torch.no_grad()
    def _get_target_single(self,
                           cls_preds,
                           decoded_bboxes,
                           mask_preds,
                           gt_bboxes,
                           gt_labels,
                           gt_masks,
                           mask_centers,
                           mask_contours,
                           gt_bboxes_ignore,
                           img_meta,
                           mlvl_num_points,
                           priors,
                           anchors):
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
            weight_targets = cls_preds.new_zeros((0,))
            return labels, bbox_targets, pos_anchors, weight_targets, pos_inds, neg_inds
        
        if self.epoch < self.initial_epoch:
            assign_result = self.static_assigner.assign(
                anchors, mlvl_num_points, gt_bboxes, mask_centers,
                mask_contours, gt_bboxes_ignore, gt_labels,
                decoded_bboxes, mask_preds, self.mask_coder)
        else:
            assign_result = self.assigner.assign(
                cls_preds.sigmoid(), decoded_bboxes, mask_preds,
                anchors, gt_bboxes, gt_masks, mask_centers, mask_contours,
                gt_bboxes_ignore, gt_labels, self.mask_coder)
            assign_metrics = assign_result.assign_metrics
            
        sampling_result = self.sampler.sample(assign_result, priors, gt_bboxes)
        labels = decoded_bboxes.new_full((priors.shape[0], ),
                                         self.num_classes,
                                         dtype=torch.long)
        label_weights = decoded_bboxes.new_ones(priors.shape[0],
                                                dtype=torch.float32)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        assign_ious = assign_result.max_overlaps
        pos_gt_inds = sampling_result.pos_assigned_gt_inds
        
        if self.epoch < self.initial_epoch:
            weight_targets = assign_ious[pos_inds]
        else:
            # ppyoloe impl
            if 'PolarMask' in self.assigner.__class__.__name__:
                am = assign_result.alignment_metrics
                norm_alignment_metrics = torch.zeros_like(am)
                norm_alignment_metrics[pos_gt_inds, pos_inds] = am[pos_gt_inds, pos_inds]
                # assert not torch.equal(alignment_metrics, alignment_metrics)

                max_metrics_per_instance, _ = norm_alignment_metrics.max(dim=1, keepdim=True)
                max_ious_per_instance, _ = assign_result.overlaps.max(dim=1, keepdim=True)
                assign_result.__delattr__('overlaps')
                assign_result.__delattr__('alignment_metrics')
                
                norm_alignment_metrics = norm_alignment_metrics / (
                    max_metrics_per_instance + 1e-9) * max_ious_per_instance
                norm_alignment_metrics, _ = norm_alignment_metrics.max(dim=0)
            else:  # ori impl
                norm_alignment_metrics = anchors.new_zeros(
                    num_points, dtype=torch.float32)
                class_assigned_gt_inds = torch.unique(pos_gt_inds)
                for gt_inds in class_assigned_gt_inds:
                    gt_class_inds = pos_inds[pos_gt_inds == gt_inds]
                    pos_alignment_metrics = assign_metrics[gt_class_inds]
                    pos_ious = assign_ious[gt_class_inds]
                    pos_norm_alignment_metrics = pos_alignment_metrics / (
                        pos_alignment_metrics.max() + 10e-8) * pos_ious.max()
                    norm_alignment_metrics[gt_class_inds] = pos_norm_alignment_metrics
            
            weight_targets = norm_alignment_metrics[pos_inds]
        
        labels[pos_inds] = gt_labels[pos_gt_inds]
        bbox_targets = sampling_result.pos_gt_bboxes
        pos_anchors = sampling_result.pos_bboxes
        pos_mask_contours = mask_contours[pos_gt_inds]
        
        return (labels, bbox_targets, pos_anchors, pos_mask_contours,
                weight_targets, pos_inds, neg_inds)

    def _varifocal_loss(self, pred_score, gt_score, label):
        """
        simple verifocal loss
        """
        assert isinstance(self.loss_cls, dict)
        alpha = self.loss_cls['alpha']
        gamma = self.loss_cls['gamma']
        loss_weight = self.loss_cls['loss_weight']
        weight_grad = self.loss_cls['weight_grad']
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        if weight_grad:
            loss = F.binary_cross_entropy_with_logits(
                pred_score, gt_score, reduction='none') * weight
            loss = loss.sum()
        else:
            loss = F.binary_cross_entropy_with_logits(
                pred_score, gt_score, weight=weight.detach(), reduction='sum')
        return loss * loss_weight

    def _get_bboxes_single(self,
                           cls_score_list,
                           pts_offsets_list,
                           score_factor_list,
                           mlvl_priors,
                           img_meta,
                           cfg,
                           rescale=False,
                           with_nms=True,
                           **kwargs):
        """Transform outputs of a single image into bbox predictions.
        """
        cfg = self.test_cfg if cfg is None else cfg
        img_shape = img_meta['img_shape']
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_masks = []
        mlvl_scores = []
        mlvl_labels = []
        for level_idx, (cls_score, pts_offset, stride, priors) in enumerate(
                zip(cls_score_list, pts_offsets_list,
                    self.prior_generator.strides, mlvl_priors)):
            assert cls_score.size()[-2:] == pts_offset.size()[-2:]
            assert stride[0] == stride[1]

            pts_offset = pts_offset.permute(1, 2, 0).reshape(
                -1, self.point_num)
            pts_offset = pts_offset * stride[0]

            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            # scores += torch.rand_like(scores)  # test

            # After https://github.com/open-mmlab/mmdetection/pull/6268/,
            # this operation keeps fewer bboxes under the same `nms_pre`.
            # There is no difference in performance for most models. If you
            # find a slight drop in performance, you can set a larger
            # `nms_pre` than before.
            results = filter_scores_and_topk(
                scores, cfg.score_thr, nms_pre,
                dict(pts_offset=pts_offset, priors=priors))
            scores, labels, _, filtered_results = results
        
            pts_offset = filtered_results['pts_offset']
            priors = filtered_results['priors']

            mask_preds = self.mask_coder.decode(
                priors[:, :2], pts_offset, max_shape=img_shape)
            
            mlvl_masks.append(mask_preds)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)

        return self._mask_post_process(
            mlvl_scores,
            mlvl_labels,
            mlvl_masks,
            img_meta['scale_factor'],
            cfg,
            rescale=rescale,
            with_nms=with_nms)

    def simple_test(self, feats, img_metas, rescale=False, instances_list=None):
        outs = self.forward(feats)
        results_list = self.get_bboxes(
            *outs, img_metas=img_metas, rescale=rescale)
        return results_list

    @force_fp32(apply_to=('cls_scores', 'pts_offsets'))
    def get_bboxes(self,
                   cls_scores,
                   pts_offsets,
                   score_factors=None,
                   img_metas=None,
                   cfg=None,
                   rescale=False,
                   with_nms=True,
                   **kwargs):
        assert len(cls_scores) == len(pts_offsets)

        if score_factors is None:
            # e.g. Retina, FreeAnchor, Foveabox, etc.
            with_score_factors = False
        else:
            # e.g. FCOS, PAA, ATSS, AutoAssign, etc.
            with_score_factors = True
            assert len(cls_scores) == len(score_factors)

        num_levels = len(cls_scores)

        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device)

        result_list = []

        for img_id in range(len(img_metas)):
            img_meta = img_metas[img_id]
            cls_score_list = select_single_mlvl(cls_scores, img_id)
            pts_offsets_list = select_single_mlvl(pts_offsets, img_id)
            if with_score_factors:
                score_factor_list = select_single_mlvl(score_factors, img_id)
            else:
                score_factor_list = [None for _ in range(num_levels)]

            results = self._get_bboxes_single(cls_score_list, pts_offsets_list,
                                              score_factor_list, mlvl_priors, img_meta,
                                              cfg, rescale, with_nms, **kwargs)
            det_bboxes, det_labels, det_masks = results
            results = InstanceData(img_meta)
            ori_shape = results.ori_shape
            
            results.bboxes = det_bboxes[:, :-1]
            results.scores = det_bboxes[:, -1]
            results.labels = det_labels
            if det_bboxes.shape[0] < 1:
                results.masks = det_bboxes.new_zeros((0, *ori_shape[:2]), dtype=torch.bool)
            else:
                im_masks = []
                masks = det_masks.detach().round().int().cpu().numpy()
                for m in masks:
                    im_mask = np.zeros(ori_shape[:2], dtype=np.uint8)
                    im_mask = cv2.drawContours(im_mask, [m], -1, 1, -1)
                    im_masks.append(im_mask)
                results.masks = torch.from_numpy(np.stack(im_masks, 0)).bool()
            result_list.append(results)
        return result_list

    def _mask_post_process(self,
                           mlvl_scores,
                           mlvl_labels,
                           mlvl_masks,
                           scale_factor,
                           cfg,
                           rescale=False,
                           with_nms=True,
                           mlvl_score_factors=None,
                           **kwargs):
        assert len(mlvl_scores) == len(mlvl_masks) == len(mlvl_labels)

        mlvl_masks = torch.cat(mlvl_masks)
        if rescale:
            mlvl_masks /= mlvl_masks.new_tensor(scale_factor)[None,None,:2]
        mlvl_scores = torch.cat(mlvl_scores)
        mlvl_labels = torch.cat(mlvl_labels)

        if mlvl_score_factors is not None:
            # TODO： Add sqrt operation in order to be consistent with
            #  the paper.
            mlvl_score_factors = torch.cat(mlvl_score_factors)
            mlvl_scores = mlvl_scores * mlvl_score_factors

        if with_nms:
            if mlvl_masks.numel() == 0:
                mlvl_bboxes = mlvl_masks.new_zeros((0, 4))
                det_bboxes = torch.cat([mlvl_bboxes, mlvl_scores[:, None]], -1)
                det_masks = det_bboxes.new_zeros((0, self.point_num))
                return det_bboxes, mlvl_labels, det_masks

            mlvl_bboxes = torch.cat([torch.min(mlvl_masks, 1)[0],
                                     torch.max(mlvl_masks, 1)[0]], dim=-1)
            det_bboxes, keep_idxs = batched_nms(mlvl_bboxes, mlvl_scores,
                                                mlvl_labels, cfg.nms)
            det_bboxes = det_bboxes[:cfg.max_per_img]
            det_masks = mlvl_masks[keep_idxs][:cfg.max_per_img]
            det_labels = mlvl_labels[keep_idxs][:cfg.max_per_img]
            return det_bboxes, det_labels, det_masks
        else:
            return mlvl_bboxes, mlvl_scores, mlvl_labels

