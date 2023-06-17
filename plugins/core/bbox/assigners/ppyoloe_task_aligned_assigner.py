# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# The code is based on:
# https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.5/ppdet/modeling/assigners/task_aligned_assigner.py
import torch
import torch.nn.functional as F

from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core.bbox.iou_calculators import build_iou_calculator
from mmdet.core.bbox.assigners.assign_result import AssignResult
from mmdet.core.bbox.assigners.base_assigner import BaseAssigner

INF = 100000000


def check_pts_in_bbox(pts, bboxes, eps=1e-9):
    assert pts.ndim == bboxes.ndim == 2
    pts_n = pts.shape[0]
    bboxes_n = bboxes.shape[0]
    # [pts_n, 1] -> [pts_n, bboxes_n]
    x, y = pts.split(1, dim=1)
    x = x.expand(-1, bboxes_n)
    y = y.expand(-1, bboxes_n)
    
    # [bboxes_n, 4] -> [1, bboxes_n]
    xmin, ymin, xmax, ymax = bboxes.t().split(1, dim=0)
    
    # [pts_n, bboxes_n]
    l = x - xmin
    t = y - ymin
    r = xmax - x
    b = ymax - y
    bbox_ltrb = torch.stack([l, t, r, b], dim=2)
    return bbox_ltrb.min(dim=-1)[0] > eps


def gather_topk_anchors(metrics, topk, largest=True, topk_mask=None, eps=1e-9):
    r"""
    Args:
        metrics (Tensor, float32): shape[B, n, L], n: num_gts, L: num_anchors
        topk (int): The number of top elements to look for along the dim.
        largest (bool) : largest is a flag, if set to true,
            algorithm will sort by descending order, otherwise sort by
            ascending order. Default: True
        topk_mask (Tensor, bool|None): shape[B, n, topk], ignore bbox mask,
            Default: None
        eps (float): Default: 1e-9
    Returns:
        is_in_topk (Tensor, float32): shape[B, n, L], value=1. means selected
    """
    num_anchors = metrics.shape[-1]
    topk_metrics, topk_idxs = torch.topk(
        metrics, topk, dim=-1, largest=largest)
    if topk_mask is None:
        topk_mask = (topk_metrics.max(dim=-1, keepdim=True) > eps).tile(
            [1, 1, topk])
    topk_idxs = torch.where(topk_mask, topk_idxs, torch.zeros_like(topk_idxs))
    is_in_topk = F.one_hot(topk_idxs, num_anchors).sum(dim=-2)
    is_in_topk = torch.where(is_in_topk > 1,
                             torch.zeros_like(is_in_topk), is_in_topk)
    return is_in_topk.astype(metrics.dtype)


@BBOX_ASSIGNERS.register_module()
class PPYOLOETaskAlignedAssigner(BaseAssigner):
    def __init__(self, topk, alpha=1, beta=6,
                 iou_calculator=dict(type='BboxOverlaps2D')):
        assert topk >= 1
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.iou_calculator = build_iou_calculator(iou_calculator)

    def assign(self,
               pred_scores,
               decode_bboxes,
               anchors,
               gt_bboxes,
               gt_bboxes_ignore=None,
               gt_labels=None):
        """Assign gt to bboxes.

        The assignment is done in following steps

        1. compute alignment metric between all bbox (bbox of all pyramid
           levels) and gt
        2. select top-k bbox as candidates for each gt
        3. limit the positive sample's center in gt (because the anchor-free
           detector only can predict positive distance)


        Args:
            pred_scores (Tensor): predicted class probability,
                shape(n, num_classes)
            decode_bboxes (Tensor): predicted bounding boxes, shape(n, 4)
            anchors (Tensor): pre-defined anchors, shape(n, 4).
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`TaskAlignedAssignResult`: The assign result.
        """
        anchors = anchors[:, :4]
        num_gt, num_bboxes = gt_bboxes.size(0), anchors.size(0)
        
        # compute alignment metric between all gtb and box[n, L]
        overlaps = self.iou_calculator(gt_bboxes, decode_bboxes).detach()
        # [cls_n, L] -> [n, L]
        bbox_scores = pred_scores.t()[gt_labels]
        # [n, L]
        alignment_metrics = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        assign_metrics = anchors.new_zeros((num_bboxes, ))
        
        if num_gt == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = anchors.new_zeros((num_bboxes, ))
            if num_gt == 0:
                # No gt boxes, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = anchors.new_full((num_bboxes, ),
                                                   -1,
                                                   dtype=torch.long)
            assign_result = AssignResult(
                num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)
            assign_result.assign_metrics = assign_metrics
            assign_result.alignment_metrics = alignment_metrics
            assign_result.overlaps = overlaps
            return assign_result
        
        points = (anchors[:, :2] + anchors[:, 2:]) / 2.0 
        # [n, L]
        is_in_gts = check_pts_in_bbox(points, gt_bboxes).t()
        
        topk = min(self.topk, alignment_metrics.size(1))
        # [n, k]
        # _, topk_ids = alignment_metrics.topk(topk, dim=1, largest=True)
        _, topk_ids = (alignment_metrics*is_in_gts).topk(topk, dim=1, largest=True)
        is_in_topk = F.one_hot(topk_ids, num_bboxes).sum(axis=-2).bool()
        mask_positive = is_in_gts & is_in_topk
        
        # if an anchor box is assigned to multiple gts,
        # the one with the highest iou will be selected, [n, L]
        mask_positive_sum = mask_positive.sum(dim=0)
        if mask_positive_sum.max() > 1:
            # [n, L] 
            mask_multiple_gts = (mask_positive_sum[None] > 1).expand(num_gt, -1)
            # [L, n] -> [n, L] 
            max_iou_index = overlaps.argmax(dim=0)
            is_max_iou = F.one_hot(max_iou_index, num_gt).bool().t()
            mask_positive = torch.where(mask_multiple_gts, is_max_iou,
                                        mask_positive)
            mask_positive_sum = mask_positive.sum(dim=0)
        
        # [L, ]
        assigned_gt_inds = mask_positive.float().argmax(dim=0)

        pos_anchor = mask_positive.any(dim=0)
        pos_anchor_inds = pos_anchor.nonzero(as_tuple=False).squeeze()
        pos_assigned_gt_inds = assigned_gt_inds[pos_anchor_inds]
        assign_metrics[pos_anchor_inds] = alignment_metrics[pos_assigned_gt_inds,
                                                            pos_anchor_inds]
        
        max_overlaps = torch.full_like(assign_metrics, -INF)
        max_overlaps[pos_anchor_inds] = overlaps[pos_assigned_gt_inds,
                                                 pos_anchor_inds]
        
        assigned_gt_inds[~pos_anchor] = -1
        assigned_gt_inds += 1
        
        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None
        assign_result = AssignResult(
            num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)
        assign_result.assign_metrics = assign_metrics
        assign_result.alignment_metrics = alignment_metrics
        assign_result.overlaps = overlaps
        return assign_result
