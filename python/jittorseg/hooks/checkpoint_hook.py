from jittorseg.utils.registry import HOOKS
from .hook import Hook
from jittorseg.utils.general import check_interval


@HOOKS.register_module()
class CheckpointHook(Hook):
    def __init__(self, runner, priority, checkpoint_interval):
        super(CheckpointHook, self).__init__(runner=runner, priority=priority)
        self.checkpoint_interval = checkpoint_interval

    def after_iter(self):
        if check_interval(self.runner.iter + 1, self.checkpoint_interval):
            self.runner.save()
