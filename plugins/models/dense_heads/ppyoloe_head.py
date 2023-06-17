# Copyright (c) OpenMMLab. All rights reserved.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule
from mmcv.cnn.utils.weight_init import normal_init, bias_init_with_prob, constant_init
from mmcv.cnn.bricks.activation import build_activation_layer
from mmcv.runner import force_fp32

from mmdet.models.dense_heads.base_dense_head import BaseDenseHead
from mmdet.models.dense_heads.dense_test_mixins import BBoxTestMixin
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.dense_heads.gfl_head import Integral
from mmdet.core import (MlvlPointGenerator, anchor_inside_flags, bbox_overlaps,
                        build_assigner, build_sampler, images_to_levels,
                        multi_apply, filter_scores_and_topk, reduce_mean,
                        unmap, distance2bbox, build_bbox_coder)


class MlvlAnchorPointGenerator(MlvlPointGenerator):
    def __init__(self, grid_cell_size, strides, offset=0.5):
        self.grid_cell_size = grid_cell_size
        super(MlvlAnchorPointGenerator, self).__init__(strides, offset)
    
    def grid_priors_anchors(self, *args, **kwargs):
        mlvl_priors = super(MlvlAnchorPointGenerator, self
                            ).grid_priors(*args, **kwargs)
        
        mlvl_anchors = []
        for stride, mp in zip(self.strides, mlvl_priors):
            shift_x, shift_y = mp[:, :2].split(1, dim=1)
            cell_half_size = self.grid_cell_size * stride[0] * 0.5
            mlvl_anchors.append(torch.cat(
                [
                    shift_x - cell_half_size, shift_y - cell_half_size,
                    shift_x + cell_half_size, shift_y + cell_half_size
                ],
                axis=1))
        return mlvl_anchors, mlvl_priors
        

class ESEAttn(nn.Module):
    def __init__(self, feat_channels,
                 norm_cfg=dict(type='BN'), act_cfg=dict(type='Swish')):
        super(ESEAttn, self).__init__()
        self.fc = nn.Conv2d(feat_channels, feat_channels, 1)
        self.conv = ConvModule(feat_channels, feat_channels, 1,
                               norm_cfg=norm_cfg, act_cfg=act_cfg)

        self._init_weights()

    def _init_weights(self):
        normal_init(self.fc.weight, std=0.001)

    def forward(self, feat, avg_feat):
        weight = self.fc(avg_feat).sigmoid()
        return self.conv(feat * weight)


class Integral_(nn.Module):
    """ppyoloe Integral
    """

    def __init__(self, reg_max=16):
        super(Integral, self).__init__()
        self.reg_max = reg_max
        # projection conv
        self.proj_conv = nn.Conv2d(self.reg_max + 1, 1, 1, bias=False)
        
        # init_weights
        self.register_buffer('project',
                             torch.linspace(0, self.reg_max, self.reg_max + 1).float())
        self.proj_conv.weight.data.copy_(
            self.project.reshape([1, self.reg_max + 1, 1, 1]))
        self.proj_conv.weight.requires_grad = False

    def forward(self, x):
        """Forward feature from the regression head to get integral result of
        bounding box location.
        """
        if self.training:
            # [?, 4]
            x = F.softmax(x.reshape(-1, 4, self.reg_max + 1), dim=1
                          ).matmul(self.project.type_as(x))
        else:
            x = x.reshape([-1, 4, self.reg_max + 1, l]).transpose(0, 2, 1, 3)
            x = self.proj_conv(F.softmax(x, dim=1))
        return x


@HEADS.register_module()
class PPYOLOEHead(BaseDenseHead, BBoxTestMixin):

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
                 use_varifocal_loss=True,
                 static_assigner=None,
                 assigner=None,
                 eval_input_size=[],
                 train_cfg=None,
                 test_cfg=None,
                 trt=False,
                 exclude_nms=False):
        super(PPYOLOEHead, self).__init__()
        assert len(in_channels) > 0, "len(in_channels) should > 0"
        in_channels = [max(round(c * width_mult), 1) for c in in_channels]
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.cls_out_channels = num_classes
        self.strides = strides
        self.grid_cell_scale = grid_cell_scale
        self.grid_cell_offset = grid_cell_offset
        self.reg_max = reg_max
        self.use_varifocal_loss = use_varifocal_loss
        self.eval_input_size = eval_input_size

        self.static_assigner = static_assigner
        self.assigner = assigner
        self.exclude_nms = exclude_nms
        
        self.act_cfg = act_cfg
        self.norm_cfg = norm_cfg
        
        if self.use_varifocal_loss:
            self.loss_cls = loss_cls
        else:
            self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_dfl = build_loss(loss_dfl)

        # self.use_l1 = False  # This flag will be modified by hooks.
        # self.loss_l1 = build_loss(loss_l1)

        self.bbox_coder = build_bbox_coder(bbox_coder)

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
                    in_c, 4 * (self.reg_max + 1), 3, padding=1))

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
            b.data.fill_(1.0)
            reg_.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            w = reg_.weight
            w.data.fill_(0.)
            reg_.weight = torch.nn.Parameter(w, requires_grad=True)
            
    def forward_single(self, feat,
                       stem_cls, pred_cls,
                       stem_reg, pred_reg):
        """Forward feature of a single scale level."""

        avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
        cls_score = pred_cls(stem_cls(feat, avg_feat) + feat)
        reg_distri = pred_reg(stem_reg(feat, avg_feat))
        return cls_score, reg_distri
    
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
    
    @force_fp32(apply_to=('cls_scores', 'reg_distri'))
    def loss(self,
             cls_scores,
             reg_distri,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        device = cls_scores[0].device
        dtype = cls_scores[0].dtype
        
        mlvl_anchors, mlvl_priors, _ = self.get_anchors(
            featmap_sizes, img_metas, dtype=dtype, device=device)
        
        cls_reg_targets = self.get_targets(
            cls_scores,
            reg_distri,
            gt_bboxes,
            gt_labels,
            gt_bboxes_ignore,
            [i[0]*i[1] for i in featmap_sizes],
            mlvl_anchors,
            mlvl_priors,
            img_metas=img_metas)
        if cls_reg_targets is None:
            return None
        
        (flatten_cls_preds, flatten_bbox_deintegral, flatten_bbox_integral, 
         labels_all, bbox_targets_all, weight_targets_all, pos_anchor_all,
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

            pos_bbox_integral = flatten_bbox_integral[pos_inds]
            pos_decode_bbox_pred = self.bbox_coder.decode(
                pos_anchor_centers, pos_bbox_integral)

            pos_decode_bbox_targets = bbox_targets_all / pos_anchor_all[:, 2:].repeat(1, 2)
            
            score[pos_inds, labels_all[pos_inds]] = weight_targets_all
            
            pred_corners = flatten_bbox_deintegral[pos_inds].reshape(-1, self.reg_max + 1)
            target_corners = self.bbox_coder.encode(pos_anchor_centers,
                                                    pos_decode_bbox_targets,
                                                    self.reg_max).reshape(-1)

            # regression loss
            losses_bbox = self.loss_bbox(
                pos_decode_bbox_pred,
                pos_decode_bbox_targets,
                weight=weight_targets_all,
                avg_factor=1.0)

            # dfl loss
            losses_dfl = self.loss_dfl(
                pred_corners,
                target_corners,
                weight=weight_targets_all[:, None].expand(-1, 4).reshape(-1),
                reduction_override='none')
            losses_dfl = losses_dfl.reshape(-1, 4).mean(dim=1).sum()

            # cls (vfl) loss
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
            losses_dfl = losses_dfl / avg_factor
            
            # if self.use_l1:
            #     losses_l1 = F.l1_loss(pos_decode_bbox_pred,
            #                           pos_decode_bbox_targets,
            #                           reduction='mean') / avg_factor
            #     return dict(
            #         loss_cls=losses_cls, losses_l1=losses_l1,
            #         loss_bbox=losses_bbox, loss_dfl=losses_dfl)
        else:
            losses_cls = sum(cls_scores.sum())*0.
            losses_bbox = sum(reg_distri.sum())*0.
            losses_dfl = sum(reg_distri.sum())*0.

        return dict(
            loss_cls=losses_cls, loss_bbox=losses_bbox, loss_dfl=losses_dfl)

    def get_targets(self, cls_scores, reg_distri, gt_bboxes_list,
                    gt_labels_list, gt_bboxes_ignore_list, mlvl_num_points,
                    mlvl_anchors, mlvl_priors, img_metas):
        num_imgs = len(img_metas)
        img_lvl_cls_preds = torch.cat([
            cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                 self.cls_out_channels)
            for cls_pred in cls_scores
        ], dim=1)
        # [b, h*w, 4*(r+1)]
        img_lvl_bbox_deintegral = torch.cat([
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4*(self.reg_max+1))
            for bbox_pred in reg_distri
        ], dim=1)
        
        # img_lvl_cls_preds = torch.cat(img_lvl_cls_preds, dim=1)
        # img_lvl_bbox_deintegral = torch.cat(img_lvl_bbox_deintegral, dim=1)
        img_lvl_bbox_integral = self.integral(
            img_lvl_bbox_deintegral).view(num_imgs, -1, 4)
        concat_priors = torch.cat(mlvl_priors, dim=0)
        concat_anchors = torch.cat(mlvl_anchors, dim=0)        # concat all level anchors and flags to a single tensor
        
        # [lvl*h*w, 2]
        flatten_centers_w_stride = \
            (concat_priors[:, :2] / concat_priors[:, 2:])[
                None].expand(num_imgs, concat_priors.shape[0], 2)

        # apply strides
        img_lvl_bbox_decode = self.bbox_coder.decode(
            flatten_centers_w_stride,
            img_lvl_bbox_integral.detach()
            ) * concat_priors[None, :, 2:].repeat(1, 1, 2)
        
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        # get labels and bbox_targets of each image
        (all_labels, all_bbox_targets, all_pos_anchors, all_weight_targets,
         pos_inds_list, neg_inds_list) = multi_apply(
            self._get_target_single,
            img_lvl_cls_preds.detach(),
            img_lvl_bbox_decode.detach(),
            gt_bboxes_list,
            gt_labels_list,
            gt_bboxes_ignore_list,
            img_metas,
            mlvl_num_points=mlvl_num_points,
            priors=concat_priors,
            anchors=concat_anchors)

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
        all_weight_targets = torch.cat(all_weight_targets, dim=0)
        all_pos_anchors = torch.cat(all_pos_anchors, dim=0)

        return (img_lvl_cls_preds, img_lvl_bbox_deintegral, img_lvl_bbox_integral,
                all_labels, all_bbox_targets, all_weight_targets, all_pos_anchors,
                num_total_pos, num_total_neg)

    @torch.no_grad()
    def _get_target_single(self,
                           cls_preds,
                           decoded_bboxes,
                           gt_bboxes,
                           gt_labels,
                           gt_bboxes_ignore,
                           img_meta,
                           mlvl_num_points,
                           priors,
                           anchors):
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
            weight_targets = cls_preds.new_zeros((0,))
            return labels, bbox_targets, pos_anchors, weight_targets, pos_inds, neg_inds
        
        if self.epoch < self.initial_epoch:
            assign_result = self.static_assigner.assign(
                anchors, mlvl_num_points, gt_bboxes,
                gt_bboxes_ignore, gt_labels, decoded_bboxes)
        else:
            assign_result = self.assigner.assign(
                cls_preds.sigmoid(), decoded_bboxes, anchors,
                gt_bboxes, gt_bboxes_ignore, gt_labels)
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
            if 'PPYOLOE' in self.assigner.__class__.__name__:
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
                class_assigned_gt_inds = torch.unique(
                    sampling_result.pos_assigned_gt_inds)
                for gt_inds in class_assigned_gt_inds:
                    gt_class_inds = pos_inds[sampling_result.pos_assigned_gt_inds ==
                                            gt_inds]
                    pos_alignment_metrics = assign_metrics[gt_class_inds]
                    pos_ious = assign_ious[gt_class_inds]
                    pos_norm_alignment_metrics = pos_alignment_metrics / (
                        pos_alignment_metrics.max() + 10e-8) * pos_ious.max()
                    norm_alignment_metrics[gt_class_inds] = pos_norm_alignment_metrics
            
            weight_targets = norm_alignment_metrics[pos_inds]
        
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        bbox_targets = sampling_result.pos_gt_bboxes
        pos_anchors = sampling_result.pos_bboxes
        
        return labels, bbox_targets, pos_anchors, weight_targets, pos_inds, neg_inds

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
                           bbox_pred_list,
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
            # scores += torch.rand_like(scores)  # test

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
