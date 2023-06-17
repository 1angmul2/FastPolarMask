from mmcv.runner.hooks import LrUpdaterHook, HOOKS


@HOOKS.register_module()
class PiceDetLrHook(LrUpdaterHook):
    def before_epoch(self, runner):
        return super().before_epoch(runner)