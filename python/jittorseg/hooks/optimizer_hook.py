from jittorseg.utils.registry import HOOKS
from .hook import Hook


@HOOKS.register_module()
class OptimizerHook(Hook):
    def __init__(self, runner, priority):
        super(OptimizerHook, self).__init__(runner=runner, priority=priority)

    def after_iter(self):
        all_loss = self.runner.outputs["all_loss"]
        self.runner.optimizer.step(all_loss)
