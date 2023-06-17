# Copyright (c) OpenMMLab. All rights reserved.
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
class YOLOMaskTaskAlignedAssigner(BaseAssigner):
    """Task aligned assigner used in the paper:
    `TOOD: Task-aligned One-stage Object Detection.
    <https://arxiv.org/abs/2108.07755>`_.

    Assign a corresponding gt bbox or background to each predicted bbox.
    Each bbox will be assigned with `0` or a positive integer
    indicating the ground truth index.

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        topk (int): number of bbox selected in each level
        iou_calculator (dict): Config dict for iou calculator.
            Default: dict(type='BboxOverlaps2D')
    """

    def __init__(self, topk, alpha=1, beta=6, mask_iou_ratio=0.5,
                 use_gt_bbox=False,
                 iou_calculator=dict(type='BboxOverlaps2D')):
        assert topk >= 1
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.mask_iou_ratio = mask_iou_ratio
        self.use_gt_bbox = use_gt_bbox
        self.iou_calculator = build_iou_calculator(iou_calculator)

    def assign(self,
               pred_scores,
               decode_bboxes,
               mask_coeffs,
               anchors,
               gt_bboxes,
               gt_masks,
               gt_bboxes_ignore=None,
               gt_labels=None,
               mask_proto=None):
        device = gt_bboxes.device
        anchors = anchors[:, :4]
        num_gt, num_bboxes = gt_bboxes.size(0), anchors.size(0)
        
        # [cls_n, L] -> [n, L]
        bbox_scores = pred_scores.t()[gt_labels]
        # [n, L]
        # 匹配系数，由此gt的类别对应的anchor的cls分数和iou共同决定
        alignment_metrics = bbox_scores.pow(self.alpha)
        assign_metrics = anchors.new_zeros((num_bboxes, ))
        
        points = (anchors[:, :2] + anchors[:, 2:]) / 2.0 

        # [n, L]
        is_in_gts = check_pts_in_bbox(points, gt_bboxes).t()
        
        gts_ids, pts_ids = is_in_gts.nonzero(as_tuple=False).t()
        
        if num_gt == 0 or num_bboxes == 0 or gts_ids.shape[0] < 1:
            # No ground truth or boxes, return empty assignment
            max_overlaps = anchors.new_zeros((num_bboxes, ))
            # No gt boxes, assign everything to background
            assigned_gt_inds = anchors.new_zeros((num_bboxes, ),
                                                 dtype=torch.long)
            assigned_labels = anchors.new_full((num_bboxes, ),
                                                -1,
                                                dtype=torch.long)
            assign_result = AssignResult(
                num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)
            assign_result.assign_metrics = assign_metrics
            assign_result.alignment_metrics = alignment_metrics
            assign_result.overlaps = anchors.new_zeros((num_gt, num_bboxes))
            return assign_result

        # gt_masks
        mask_feat_size = mask_proto.shape[-2:]
        stride = 4.
        upsampled_size = (mask_feat_size[0]*stride, mask_feat_size[1]*stride)
        b_pad = upsampled_size[0] - gt_masks.shape[-2]
        r_pad = upsampled_size[1] - gt_masks.shape[-1]
        assert b_pad>=0 and r_pad>=0
        if not ((b_pad==0) and (r_pad==0)):
            gt_masks_pad = F.pad(gt_masks, [0, r_pad, 0, b_pad],
                                mode='constant', value=0)
        else:
            gt_masks_pad = gt_masks
        gt_masks = F.interpolate(gt_masks_pad.unsqueeze(1).float(),
                                scale_factor=1. / stride,
                                mode='bilinear', align_corners=False).squeeze(1)
        # gt_bboxes_ws = decode_bboxes[_p] / stride
        # gt_masks = crop_mask(gt_masks, gt_bboxes_ws)

        limit_num = 200
        condidate_iou = points.new_zeros((len(gts_ids), ))
        _gts_ids = torch.split(gts_ids, limit_num)
        _pts_ids = torch.split(pts_ids, limit_num)
        _ids = torch.split(torch.arange(len(gts_ids), device=device), limit_num)
        for _i, _g, _p in zip(_ids, _gts_ids, _pts_ids):
            # [n, 32]
            condidate_mask_coeff = mask_coeffs[_p]
            # [n, h, w]
            condidate_mask_preds = torch.einsum(
                'chw, nc -> nhw', mask_proto, condidate_mask_coeff).sigmoid()
            if self.use_gt_bbox:
                condidate_decode_bboxes = gt_bboxes[_g] / stride  # down_stride
            else:
                condidate_decode_bboxes = decode_bboxes[_p] / stride  # down_stride
            condidate_gt_masks = gt_masks[_g]
            condidate_mask_preds = crop_mask(
                condidate_mask_preds, condidate_decode_bboxes)
            # condidate_gt_masks = crop_mask(
            #     condidate_gt_masks, condidate_decode_bboxes)
            condidate_iou[_i] = mask_iou(condidate_mask_preds, condidate_gt_masks)
        
        # [pos_n, ]
        overlaps = anchors.new_zeros((num_gt, num_bboxes))
        overlaps[gts_ids, pts_ids] = condidate_iou
        # compute alignment metric between all gtb and box[n, L]
        overlaps_bbox = self.iou_calculator(gt_bboxes, decode_bboxes).detach()
        overlaps = overlaps * self.mask_iou_ratio + overlaps_bbox * (1-self.mask_iou_ratio)
        alignment_metrics *= overlaps.pow(self.beta)
        
        topk = min(self.topk, alignment_metrics.size(1))
        # [n, k]
        # _, topk_ids = alignment_metrics.topk(topk, dim=1, largest=True)
        # 计算每个gt的前13大匹配系数的索引
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
        pos_anchor_inds = pos_anchor.nonzero(as_tuple=False).squeeze(1)
        pos_assigned_gt_inds = assigned_gt_inds[pos_anchor_inds]
        assign_metrics[pos_anchor_inds] = alignment_metrics[pos_assigned_gt_inds,
                                                            pos_anchor_inds]
        
        max_overlaps = torch.full_like(assign_metrics, -INF)
        max_overlaps[pos_anchor_inds] = overlaps[pos_assigned_gt_inds,
                                                 pos_anchor_inds]
        
        assigned_gt_inds[~pos_anchor] = -1
        assigned_gt_inds += 1
        
        assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
        pos_inds = torch.nonzero(
            assigned_gt_inds > 0, as_tuple=False).squeeze()
        if pos_inds.numel() > 0:
            assigned_labels[pos_inds] = gt_labels[
                assigned_gt_inds[pos_inds] - 1]

        assign_result = AssignResult(
            num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)
        assign_result.assign_metrics = assign_metrics
        assign_result.alignment_metrics = alignment_metrics
        assign_result.overlaps = overlaps
        return assign_result


def crop_mask(masks, boxes):
    """
    "Crop" predicted masks by zeroing out everything not in the predicted bbox.
    Vectorized by Chong (thanks Chong).

    Args:
        - masks should be a size [h, w, n] tensor of masks
        - boxes should be a size [n, 4] tensor of bbox coords in relative point form
    """

    n, h, w = masks.shape
    x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)  # x1 shape(1,1,n)
    r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]  # rows shape(1,w,1)
    c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]  # cols shape(h,1,1)

    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


def mask_iou(preds, gts):
    # [n, h, w]
    inter = (preds * gts).sum([1, 2])
    area_preds = preds.sum([1, 2]).clamp(min=1e-3)
    area_gts = gts.sum([1, 2]).clamp(min=1e-3)
    iou = inter / (area_preds + area_gts - inter)
    return iou

