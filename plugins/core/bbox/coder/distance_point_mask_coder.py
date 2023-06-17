# Copyright (c) OpenMMLab. All rights reserved.
import torch
import numpy as np

from mmdet.core.bbox.builder import BBOX_CODERS
from mmdet.core.bbox.coder.base_bbox_coder import BaseBBoxCoder
from ...utils.point_polygon import points_in_polygon


@BBOX_CODERS.register_module()
class DistancePointMaskCoder(BaseBBoxCoder):
    """
    polarmask
    """
    def __init__(self, point_num=36, dist_k=3, clip_border=True, mem_limit=0):
        super(BaseBBoxCoder, self).__init__()
        self.point_num = point_num
        self.dist_k = dist_k
        self.clip_border = clip_border
        self.mem_limit=mem_limit

    def encode(self, points, contours, max_dis=None, eps=0.1, pass_plg=False):
        assert points.size(0) == contours.size(0)
        assert points.size(-1) == 2
        assert contours.size(-1) == 2
        
        if pass_plg:
            pts_in_plg = points.new_ones((points.size(0), ), dtype=torch.bool)
        else:
            pts_in_plg = points_in_polygon(points, contours, True, self.mem_limit)
        
        m = points.shape[0]
        n = contours.shape[1]
        num_chunks = 1
        if self.mem_limit > 0:
            # [m, n, 2]
            num_chunks = int(
                np.ceil(m*self.point_num*n*6*4 / (self.mem_limit*1024**3)))
        if num_chunks > 1:
            device = points.device
            chunks = torch.chunk(torch.arange(m, device=device), num_chunks)
            out = points.new_zeros((m, self.point_num))
            for inds in chunks:
                out[inds] = mask2distance(
                    points[inds], contours[inds], pts_in_plg[inds],
                    self.dist_k, self.point_num, max_dis, eps)
        else:
            out = mask2distance(points, contours, pts_in_plg, self.dist_k,
                                self.point_num, max_dis, eps)
        return out

    def decode(self, points, offsets, max_shape=None):
        assert points.size(0) == offsets.size(0)
        assert points.size(-1) == 2
        assert offsets.size(-1) == self.point_num
        if self.clip_border is False:
            max_shape = None
        # [n, pn, 2]
        return distance2mask(
            points, offsets, self.point_num, max_shape)


def distance2mask(points, offsets, point_num, max_shape=None):
    # [b, n, 2]
    c_x, c_y = torch.split(points, 1, -1)

    angles = torch.arange(
        0, 360, 360//point_num,
        device=points.device,
        dtype=points.dtype) * np.pi / 180
    if points.ndim == 3:
        sin = torch.sin(angles)[None, None]
        cos = torch.cos(angles)[None, None]
    elif points.ndim == 2:
        sin = torch.sin(angles)[None]
        cos = torch.cos(angles)[None]
    else:
        raise NotImplementedError

    # [b, n, pn]
    x = offsets * cos + c_x
    y = offsets * sin + c_y

    if max_shape is not None:
        x = x.clamp(min=0, max=max_shape[1])
        y = y.clamp(min=0, max=max_shape[0])

    # [b, n, pn, 2]
    return torch.stack([x, y], dim=-1)


def mask2distance(pos_cts, pos_contours, pts_in_plg,
                  k, point_num, max_dis, eps):
    pn = point_num
    # [m, n]
    angle_per_ct = get_angle(pos_cts, pos_contours)
    
    # [m, pn, n]
    theta = torch.arange(0, 360, 360//pn).type_as(pos_cts)
    
    # [m, pn, n]
    # 计算角度差，并解决 diff=359-1 这种角度问题
    diff_a = (angle_per_ct[:, None] - theta[None, :, None]).abs_()
    diff_a = torch.where(diff_a.gt(180.), 360 - diff_a, diff_a)
    
    # 选角度差异最小的topk位置，选择其中距离最长的为 期望角度 的距离
    # [m, pn, k]
    val_a, idx_a = diff_a.topk(k , 2, largest=False)
    del diff_a
    # [m, n] -> [m, pn, n] -> [m, pn, n]
    dist = torch.norm(pos_contours - pos_cts[:, None], 2, 2)
    # dist = ((pos_contours - pos_cts[:, None])**2).sum(2)
    dist = dist[:, None].expand(-1, pn, -1)
    # 注意gather的用法，可以直接根据topk索引对应的值
    dist_a = torch.gather(dist, 2, idx_a)
    
    # 这个版本有可能pts在box里面但是不再conts里面
    # 所以对于点在conts外面的，那些最小角度大于2度的为0,因为射线没有负的
    val_bool = ((~pts_in_plg[...,None,None]) & val_a.min(2)[0].gt(2)[...,None]
                ).expand(-1,-1,k)
    dist_a = torch.where(val_bool,
                         torch.full_like(dist_a, 1e-6),
                         dist_a)
    
    # 取topk中最大的距离即为想要的 距离GT，[m, pst_n]
    dist_max, dist_ids = dist_a.max(2)
    
    if max_dis is not None:
        return dist_max.clamp(min=1e-6, max=max_dis-eps)
    else:
        return dist_max.clamp(min=1e-6)


def get_angle(ct_pts, conts):
    """
    ct_pts[m, 2]
    conts[m, n, 2]
    计算m个中心点与n个定点围成的角度
    输出：[m, n]
    """

    # conts->ct_pts
    # [m, n, 2]
    v = conts - ct_pts[:, None]
    
    # [m, n]
    # 与x轴正方向的夹角，顺时针为正,atan2输出为(-180,180]
    angle = torch.atan2(v[..., 1], v[..., 0])
    angle = angle * 180. / np.pi
    included_angle = torch.where(
        angle<0, angle+360, angle)
    return included_angle
