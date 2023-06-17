# Copyright (c) OpenMMLab. All rights reserved.
import torch
import numpy as np
import torch.nn.functional as F

from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core.bbox.iou_calculators import build_iou_calculator
from mmdet.core.bbox.assigners.assign_result import AssignResult
from mmdet.core.bbox.assigners.base_assigner import BaseAssigner

from ...utils.point_polygon import mask_iou


@BBOX_ASSIGNERS.register_module()
class PolarMaskATSSAssigner(BaseAssigner):
    def __init__(self,
                 topk,
                 iou_calculator=dict(type='BboxOverlaps2D'),
                 use_mask_cts=False,
                 use_mask_iou=True,
                 use_anchor=True,
                 ignore_iof_thr=-1):
        self.topk = topk
        self.use_mask_cts = use_mask_cts
        self.use_mask_iou = use_mask_iou
        self.use_anchor = use_anchor
        self.iou_calculator = build_iou_calculator(iou_calculator)
        self.ignore_iof_thr = ignore_iof_thr

    def assign(self,
               bboxes,
               num_level_bboxes,
               gt_bboxes,
               mask_centers,
               mask_contours,
               gt_bboxes_ignore=None,
               gt_labels=None,
               decoded_bboxes=None,
               mask_preds=None,
               mask_coder=None):
        device = gt_bboxes.device
        INF = 100000000
        bboxes = bboxes[:, :4]
        num_gt, num_bboxes = gt_bboxes.size(0), bboxes.size(0)

        # compute iou between all bbox and gt
        overlaps = self.iou_calculator(bboxes, gt_bboxes)

        # assign 0 by default
        assigned_gt_inds = overlaps.new_full((num_bboxes, ),
                                             0,
                                             dtype=torch.long)

        if num_gt == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_bboxes, ))
            if num_gt == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_full((num_bboxes, ),
                                                    -1,
                                                    dtype=torch.long)
            return AssignResult(
                num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

        if self.use_mask_cts:
            gt_points = mask_centers
        else:
            # compute center distance between all bbox and gt
            gt_cx = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2.0
            gt_cy = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2.0
            gt_points = torch.stack((gt_cx, gt_cy), dim=1)

        bboxes_cx = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
        bboxes_cy = (bboxes[:, 1] + bboxes[:, 3]) / 2.0
        bboxes_points = torch.stack((bboxes_cx, bboxes_cy), dim=1)

        distances = torch.norm(bboxes_points[:, None, :] - gt_points[None, :, :],
                               p=2, dim=-1)

        if (self.ignore_iof_thr > 0 and gt_bboxes_ignore is not None
                and gt_bboxes_ignore.numel() > 0 and bboxes.numel() > 0):
            ignore_overlaps = self.iou_calculator(
                bboxes, gt_bboxes_ignore, mode='iof')
            ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            ignore_idxs = ignore_max_overlaps > self.ignore_iof_thr
            distances[ignore_idxs, :] = INF
            assigned_gt_inds[ignore_idxs] = -1

        # Selecting candidates based on the center distance
        candidate_idxs = []
        start_idx = 0
        for level, bboxes_per_level in enumerate(num_level_bboxes):
            # on each pyramid level, for each gt,
            # select k bbox whose center are closest to the gt center
            end_idx = start_idx + bboxes_per_level
            distances_per_level = distances[start_idx:end_idx, :]
            selectable_k = min(self.topk, bboxes_per_level)
            _, topk_idxs_per_level = distances_per_level.topk(
                selectable_k, dim=0, largest=False)
            candidate_idxs.append(topk_idxs_per_level + start_idx)
            start_idx = end_idx
        candidate_idxs = torch.cat(candidate_idxs, dim=0)

        # get corresponding iou for the these candidates, and compute the
        # mean and std, set mean + std as the iou threshold
        candidate_overlaps = overlaps[candidate_idxs, torch.arange(num_gt)]
        overlaps_mean_per_gt = candidate_overlaps.mean(0)
        overlaps_std_per_gt = candidate_overlaps.std(0)
        overlaps_thr_per_gt = overlaps_mean_per_gt + overlaps_std_per_gt

        is_pos = candidate_overlaps >= overlaps_thr_per_gt[None, :]

        # limit the positive sample's center in gt
        for gt_idx in range(num_gt):
            candidate_idxs[:, gt_idx] += gt_idx * num_bboxes
        ep_bboxes_cx = bboxes_cx.view(1, -1).expand(
            num_gt, num_bboxes).contiguous().view(-1)
        ep_bboxes_cy = bboxes_cy.view(1, -1).expand(
            num_gt, num_bboxes).contiguous().view(-1)
        candidate_idxs = candidate_idxs.view(-1)

        # calculate the left, top, right, bottom distance between positive
        # bbox center and gt side
        l_ = ep_bboxes_cx[candidate_idxs].view(-1, num_gt) - gt_bboxes[:, 0]
        t_ = ep_bboxes_cy[candidate_idxs].view(-1, num_gt) - gt_bboxes[:, 1]
        r_ = gt_bboxes[:, 2] - ep_bboxes_cx[candidate_idxs].view(-1, num_gt)
        b_ = gt_bboxes[:, 3] - ep_bboxes_cy[candidate_idxs].view(-1, num_gt)
        is_in_gts = torch.stack([l_, t_, r_, b_], dim=1).min(dim=1)[0] > 0.01
        is_pos = is_pos & is_in_gts

        # if an anchor box is assigned to multiple gts,
        # the one with the highest IoU will be selected.
        overlaps_inf = torch.full_like(overlaps,
                                       -INF).t().contiguous().view(-1)
        index = candidate_idxs.view(-1)[is_pos.view(-1)]
        overlaps_inf[index] = overlaps.t().contiguous().view(-1)[index]
        overlaps_inf = overlaps_inf.view(num_gt, -1).t()

        max_overlaps, argmax_overlaps = overlaps_inf.max(dim=1)
        pos_idx = max_overlaps != -INF
        assigned_gt_inds[pos_idx] = argmax_overlaps[pos_idx] + 1

        assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
        pos_inds = torch.nonzero(
            assigned_gt_inds > 0, as_tuple=False).squeeze(1)
        if pos_inds.numel() > 0:
            assigned_labels[pos_inds] = gt_labels[
                assigned_gt_inds[pos_inds] - 1]
        
        # 增加计算pred boxes和anchor的iou
        if decoded_bboxes is not None or mask_preds is not None:
            if len(pos_inds) < 1:
                max_overlaps = overlaps_inf.new_full((num_bboxes, ), -INF)
            else:
                if self.use_mask_iou:
                    mask_preds = mask_preds[pos_inds]
                    contours = mask_contours[assigned_gt_inds[pos_inds] - 1]
                    _bboxes_points = bboxes_points[pos_inds]
                    gt_mask_encode = mask_coder.encode(_bboxes_points, contours)
                    pred_overlaps = mask_iou(mask_preds, gt_mask_encode)
                    max_overlaps = pred_overlaps.new_full((num_bboxes, ), -INF)
                    max_overlaps[pos_inds] = pred_overlaps
                else:
                    pred_overlaps = self.iou_calculator(gt_bboxes, decoded_bboxes)
                    # max_overlaps[pos_idx] = pred_overlaps.view(-1)[index]
                    max_overlaps[pos_inds] = pred_overlaps[assigned_gt_inds[pos_inds] - 1, pos_inds]
        
        return AssignResult(
            num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)


def compute_max_iou_anchor(ious):
    r"""
    For each anchor, find the GT with the largest IOU.
    Args:
        ious (Tensor, float32): shape[n, L], n: num_gts, L: num_anchors
    Returns:
        is_max_iou (Tensor, float32): shape[n, L], value=1. means selected
    """
    num_max_boxes = ious.shape[-2]
    max_iou_index = ious.argmax(dim=-2)
    is_max_iou = F.one_hot(max_iou_index, num_max_boxes).t()
    return is_max_iou


def compute_max_iou_gt(ious):
    r"""
    For each GT, find the anchor with the largest IOU.
    Args:
        ious (Tensor, float32): shape[n, L], n: num_gts, L: num_anchors
    Returns:
        is_max_iou (Tensor, float32): shape[n, L], value=1. means selected
    """
    num_anchors = ious.shape[-1]
    max_iou_index = ious.argmax(dim=-1)
    is_max_iou = F.one_hot(max_iou_index, num_anchors)
    return is_max_iou
