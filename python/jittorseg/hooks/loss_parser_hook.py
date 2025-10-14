from jittorseg.utils.registry import HOOKS
from .hook import Hook
from jittorseg.utils.general import parse_losses


@HOOKS.register_module()
class LossParserHook(Hook):
    def __init__(self, runner, priority):
        super(LossParserHook, self).__init__(runner=runner, priority=priority)

    def after_iter(self):
        all_loss, loss = parse_losses(self.runner.outputs["losses"])
        self.runner.outputs["all_loss"] = all_loss
        self.runner.outputs["loss"] = loss
