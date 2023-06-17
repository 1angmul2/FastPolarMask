import torch
import numpy as np


def points_in_polygon(pts, polygon, align=False, mem_limit=0):
    if align:
        return points_in_polygon_align(pts, polygon)
    m = pts.shape[0]
    k, n = polygon.shape[:2]
    num_chunks = 1
    if mem_limit > 0:
        # [m, k, n, 2]
        num_chunks = int(
            np.ceil(m*k*n*6*4 / (mem_limit*1024**3)))
    if num_chunks > 1:
        device = pts.device
        chunks = torch.chunk(torch.arange(m, device=device), num_chunks)
        out = pts.new_zeros((m, k), dtype=torch.bool)
        for inds in chunks:
            out[inds] = _points_in_polygon(pts[inds], polygon)
    else:
        out = _points_in_polygon(pts, polygon)
    return out


def _points_in_polygon(pts, polygon):
    """
    pts[m, 2]
    polygon[k, n, 2]
    return [m, k]
    """
    # [m, 2] -> [m, k, n, 2]
    pts = pts[:, None, None, :]
    # [k, n, 2] -> [m, k, n, 2]
    polygon = polygon[None]
    
    # roll, [k, n, 2]
    contour2 = torch.roll(polygon, -1, -2)
    # 计算边向量
    test_diff = contour2 - polygon
    # [m, k, n, 2] -> [m, k]
    mask1 = (pts == polygon).all(-1).any(-1)
    # [m, k, n]
    m1 = (polygon[...,1] > pts[...,1]) != (contour2[...,1] > pts[...,1])
    slope = ((pts[...,0]-polygon[...,0])*test_diff[...,1])-(
             test_diff[...,0]*(pts[...,1]-polygon[...,1]))
    m2 = slope == 0
    mask2 = (m1 & m2).any(-1)
    m3 = (slope < 0) != (contour2[...,1] < polygon[...,1])
    m4 = m1 & m3
    count = torch.count_nonzero(m4, dim=-1)
    mask3 = ~(count%2==0)
    mask = mask1 | mask2 | mask3
    return mask


def points_in_polygon_align(pts, polygon):
    """
    pts[m, 2]
    polygon[m, n, 2]
    return [m, ]
    """
    # roll
    contour2 = torch.roll(polygon, -1, 1)
    test_diff = contour2 - polygon
    # [m, n, 2] -> [m, ]
    mask1 = (pts[:,None] == polygon).all(-1).any(-1)
    # [n, 2]
    m1 = (polygon[...,1] > pts[:,None,1]) != (contour2[...,1] > pts[:,None,1])
    slope = ((pts[:,None,0]-polygon[...,0])*test_diff[...,1])-(
             test_diff[...,0]*(pts[:,None,1]-polygon[...,1]))
    m2 = slope == 0
    mask2 = (m1 & m2).any(-1)
    m3 = (slope < 0) != (contour2[...,1] < polygon[...,1])
    m4 = m1 & m3
    count = torch.count_nonzero(m4, dim=-1)
    mask3 = ~(count%2==0)
    mask = mask1 | mask2 | mask3
    return mask


def mask_iou(pred, target, align=True):
    """
    pred[m,pn]
    target[n?,pn]
    """
    if align:
        total = torch.stack([pred,target], -1)
        l_max = total.max(dim=2)[0]
        l_min = total.min(dim=2)[0]
    else:
        l_min = torch.minimum(pred[:,None], target[None])
        l_max = torch.maximum(pred[:,None], target[None])
    return l_min.sum(dim=-1) / l_max.sum(dim=-1)

