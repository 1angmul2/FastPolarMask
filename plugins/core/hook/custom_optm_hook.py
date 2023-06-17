# Copyright (c) OpenMMLab. All rights reserved.
from cmath import isnan
from mmcv.parallel import is_module_wrapper
from mmcv.runner import HOOKS, OptimizerHook, TensorboardLoggerHook
from mmcv.runner.dist_utils import master_only
import torch
import os


@HOOKS.register_module()
class GradRecordOptimizerHook(OptimizerHook):
    """Set runner's epoch information to the model."""
    def __init__(self, grad_clip=None, detect_anomalous_params=False,
                 interval=None):
        self.grad_clip = grad_clip
        self.detect_anomalous_params = detect_anomalous_params
        self.tb_hook = None
        self.interval = interval

    def after_train_iter(self, runner):
        if torch.isnan(runner.outputs['loss']):
            print(self.get_tb_hook(runner).get_iter(runner), 'loss_nan')
        
        runner.optimizer.zero_grad()
        if self.detect_anomalous_params:
            self.detect_anomalous_parameters(runner.outputs['loss'], runner)
        runner.outputs['loss'].backward()

        self.grad_log('bw_only', runner, _grad_=False, _param_=False)
        # self.grad_log('bw_only', runner, _grad_=True, _param_=True)
        # self.grad_log('bw_only', runner, _grad_=False, _param_=True)
        # self.grad_log('bw_only', runner, _grad_=True, _param_=False)

        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters())
            
            self.grad_log('af_clip', runner, _grad_=False, _param_=True)
            # self.grad_log('af_clip', runner, _grad_=True, _param_=True)
            # self.grad_log('af_clip', runner, _grad_=False, _param_=True)
            # self.grad_log('af_clip', runner, _grad_=True, _param_=False)
            
            if grad_norm is not None:
                # Add grad norm to the logger
                runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                         runner.outputs['num_samples'])
        runner.optimizer.step()
        self.grad_log('af_step', runner, _grad_=False, _param_=False)
        # self.grad_log('af_step', runner, _grad_=True, _param_=True)
        # self.grad_log('af_step', runner, _grad_=False, _param_=True)
        # self.grad_log('af_step', runner, _grad_=True, _param_=False)
        
    def get_tb_hook(self, runner):
        if self.tb_hook is None:
            for h in runner._hooks[::-1]:
                if isinstance(h, TensorboardLoggerHook):
                    self.tb_hook = h
                    self.by_epoch = self.tb_hook.by_epoch
                    self.ignore_last = self.tb_hook.by_epoch
                    if self.interval is None:
                        self.interval = self.tb_hook.interval
        assert self.tb_hook is not None
        return self.tb_hook
    
    def get_model(self, runner):
        if is_module_wrapper(runner.model):
            model = runner.model.module
        else:
            model = runner.model
        return model
    
    def get_log_bool(self, runner):
        # ref mmcv/runner/hooks/logger/base.py
        log_bool = False
        if self.by_epoch and self.every_n_inner_iters(runner, self.interval):
            log_bool = True
        elif not self.by_epoch and self.every_n_iters(runner, self.interval):
            log_bool = True
        elif self.end_of_epoch(runner) and not self.ignore_last:
            # not precise but more stable
            log_bool = True
        return log_bool
    
    @master_only
    def grad_log(self, log_name, runner, _grad_=True, _param_=False):
        log_name = " -[ {} ]-: ".format(log_name)
        tb_hook = self.get_tb_hook(runner)
        model = self.get_model(runner)
        
        if self.get_log_bool(runner):
            this_iter = tb_hook.get_iter(runner)
            for name, param in model.named_parameters():
                    
                with open(os.path.join(tb_hook.log_dir, 'test_param.txt'), 'a') as f:
                    print(this_iter,"-"*10,log_name+name,"-"*10, param.max().item(), file=f)
                with open(os.path.join(tb_hook.log_dir, 'test_grad.txt'), 'a') as f:
                    print(this_iter,"-"*10,log_name+name,"-"*10, param.grad.max().item(), file=f)
                    
                if _grad_:
                    tb_hook.writer.add_histogram(log_name + name + '_grad', param.grad, this_iter)
                if _param_:
                    tb_hook.writer.add_histogram(log_name + name + '_param', param, this_iter)
