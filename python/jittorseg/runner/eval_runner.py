from jittorseg.utils.registry import build_from_cfg, DATASETS
from .base_runner import BaseRunner
from jittorseg.hooks import Priority


class EvalRunner(BaseRunner):

    def __init__(self):
        super().__init__()
        self.val_dataset = build_from_cfg(self.cfg.dataset.val, DATASETS)
        self.register_hook(dict(type="EvalHook", priority=Priority.NORMAL, eval_interval=1, efficient_val=self.cfg.efficient_val))

    def run(self):
        self.call_hook("after_iter")


