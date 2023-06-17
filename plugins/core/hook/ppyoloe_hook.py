# Copyright (c) OpenMMLab. All rights reserved.
import math
import torch
from mmcv.parallel import is_module_wrapper
from mmcv.runner.hooks import HOOKS, Hook
from mmdet.core.hook.ema import BaseEMAHook


@HOOKS.register_module()
class PPYOLOEExpMomentumEMAHook(BaseEMAHook):
    """EMAHook using exponential momentum strategy.

    Args:
        total_iter (int): The total number of iterations of EMA momentum.
           Defaults to 2000.
    """

    def __init__(self, total_iter=2000, **kwargs):
        super(PPYOLOEExpMomentumEMAHook, self).__init__(**kwargs)
        self.momentum_fun = lambda x: (1 - self.momentum) * math.exp(-(
            1 + x) / total_iter)


@HOOKS.register_module()
class PPYOLOETrunOnSybnHook(Hook):
    """turn on sybn"""

    def before_run(self, runner):
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
            torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        else:
            torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
