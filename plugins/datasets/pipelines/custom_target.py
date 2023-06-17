# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
import cv2
import mmcv
import numpy as np
from numpy import random
from scipy.interpolate import interp1d
from numba import njit
from loguru import logger
from mmcv.parallel import DataContainer as DC

from mmdet.core import mask2ndarray
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import DefaultFormatBundle, to_tensor



@PIPELINES.register_module()
class PolarMaskTarget(object):
    def __init__(self, point_num, oversampling_rate=50,
                 oversampling_mode='linear'):
        self.point_num = point_num
        self.oversampling_rate = oversampling_rate
        self.oversampling_mode = oversampling_mode
    
    # @logger.catch()
    def __call__(self, results):
        gt_masks = mask2ndarray(results['gt_masks'])
        mask_centers = []
        mask_contours = []
        #[num_gt, 2]
        for mask in gt_masks:
            cnt, contour = self.get_single_centerpoint(mask)
            mask_centers.append(cnt)
            mask_contours.append(contour)
            
        if len(mask_contours) < 1:
            return None
        
        mask_centers = np.array(mask_centers, dtype=np.float32).copy()
        mask_contours = np.stack(mask_contours, 0).astype(np.float32).copy()
        
        # TODO: fix
        if True:
            gt_bboxes = np.concatenate(
                [np.min(mask_contours, -2),
                np.max(mask_contours, -2)], axis=-1)
            results['gt_bboxes'] = gt_bboxes.copy()
        
        results['mask_centers'] = mask_centers
        results['mask_contours'] = mask_contours
        results['bbox_fields'].append('mask_centers')
        results['bbox_fields'].append('mask_contours')
        return results

    def get_single_centerpoint(self, mask):
        # [n, 2] -> [m, n, 2]
        contour, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)# cv2.RETR_TREE
        
        # if len(contour) == 0:
        #     return None
        
        contour = sorted(contour, key=lambda x: cv2.contourArea(x), reverse=True)  # only save the biggest one

        '''debug IndexError: list index out of range'''
        contour = contour[0][:, 0, :]
        try:
            center = get_centerpoint(contour)
        except:
            x,y = contour.mean(axis=0)
            center=(int(x), int(y))
        
        point_num = self.oversampling_rate * self.point_num
        length = len(contour)
        if length != point_num:
            contour = oversampling_contour(
                contour, point_num, mode=self.oversampling_mode)
        assert len(contour) == point_num
        return center, contour


@njit
def get_centerpoint(lis):
    area = 0.0
    x, y = 0.0, 0.0
    a = len(lis)
    for i in range(a):
        lat = lis[i][0]
        lng = lis[i][1]
        if i == 0:
            lat1 = lis[-1][0]
            lng1 = lis[-1][1]
        else:
            lat1 = lis[i - 1][0]
            lng1 = lis[i - 1][1]
        fg = (lat * lng1 - lng * lat1) / 2.0
        area += fg
        x += fg * (lat + lat1) / 3.0
        y += fg * (lng + lng1) / 3.0
    x = x / area
    y = y / area
    return float(x), float(y)


def oversampling_contour(contour, point_num, colsed=False, mode='linear'):
    """插值"""
    if not colsed:
        contour = np.concatenate(
            [contour, contour[[0]]], 0)
    # rate = point_num / len(contour)
    # 100,99,100/99
    x, y = contour.T
    length = len(contour)
    ids = np.arange(length)

    # 5x the original number of points
    interp_i = np.linspace(0, length-1, point_num)

    xi = interp1d(ids, x, kind=mode)(interp_i)
    yi = interp1d(ids, y, kind=mode)(interp_i)
    return np.stack([xi, yi], axis=1)


@PIPELINES.register_module()
class PolarMaskFormatBundle(DefaultFormatBundle):
    def __call__(self, results):
        if 'img' in results:
            img = results['img']
            if self.img_to_float is True and img.dtype == np.uint8:
                # Normally, image is of uint8 type without normalization.
                # At this time, it needs to be forced to be converted to
                # flot32, otherwise the model training and inference
                # will be wrong. Only used for YOLOX currently .
                img = img.astype(np.float32)
            # add default meta keys
            results = self._add_default_meta_keys(results)
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            results['img'] = DC(
                to_tensor(img), padding_value=self.pad_val['img'], stack=True)
        for key in ['proposals', 'gt_bboxes', 'gt_bboxes_ignore',
                    'gt_labels', 'mask_centers', 'mask_contours']:
            if key not in results:
                continue
            results[key] = DC(to_tensor(results[key].copy()))
        if 'gt_masks' in results:
            results['gt_masks'] = DC(
                to_tensor(results['gt_masks'].to_ndarray().copy()),
                padding_value=self.pad_val['masks']
                )
        if 'gt_semantic_seg' in results:
            results['gt_semantic_seg'] = DC(
                to_tensor(results['gt_semantic_seg'][None, ...].copy()),
                padding_value=self.pad_val['seg'],
                stack=True)
        return results



@PIPELINES.register_module()
class PolarMaskCoderDebugShow:
    def __init__(self, point_num=36):
        self.point_num=point_num
        
    # @logger.catch()
    def __call__(self, results):
        import torch
        from mmdet.plugins.core.bbox.coder import DistancePointMaskCoder
        from mmdet.core.mask.structures import BitmapMasks
        coder = DistancePointMaskCoder(self.point_num, 3)
        cts = torch.from_numpy(results['mask_centers']).float()
        conts = torch.from_numpy(results['mask_contours']).float()
        mask_encode = coder.encode(cts, conts)
        mask_decode = coder.decode(cts, mask_encode)
        im_masks = []
        masks = mask_decode.detach().round().int().cpu().numpy()  # decode
        # masks = conts.detach().round().int().cpu().numpy()  # label
        shape = (results['gt_masks'].height, results['gt_masks'].width)
        for m in masks:
            im_mask = np.zeros(shape[:2], dtype=np.uint8)
            im_mask = cv2.drawContours(im_mask, [m], -1, 1, -1)
            im_masks.append(im_mask)
        masks = np.stack(im_masks, 0).astype(np.int)
        masks = BitmapMasks(masks, shape[0], shape[1])
        results['gt_polarmask'] = masks
        return results
