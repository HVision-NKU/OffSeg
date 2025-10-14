from jittorseg.utils.registry import HOOKS
from .hook import Hook


@HOOKS.register_module()
class LRSchedulerHook(Hook):
    def __init__(self, runner, priority, by_epoch):
        super(LRSchedulerHook, self).__init__(runner=runner, priority=priority)
        self.by_epoch = by_epoch

    def after_iter(self):
        self.runner.scheduler.step(self.runner.iter, self.runner.epoch, by_epoch=self.by_epoch)
