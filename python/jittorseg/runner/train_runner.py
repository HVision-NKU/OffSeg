import jittor as jt
from jittorseg.utils.registry import build_from_cfg, DATASETS
from .base_runner import BaseRunner
from jittorseg.hooks import Priority


class TrainRunner(BaseRunner):

    def __init__(self):
        super().__init__()
        self.train_dataset = build_from_cfg(self.cfg.dataset.train,
                                            DATASETS,
                                            drop_last=jt.in_mpi)
        self.val_dataset = build_from_cfg(self.cfg.dataset.val, DATASETS)

        if self.max_epoch:
            if (self.train_dataset):
                self.total_iter = self.max_epoch * len(self.train_dataset)
            else:
                self.total_iter = 0
        else:
            self.total_iter = self.max_iter

        self.register_train_hooks()
        self.register_custom_hooks()
        # if jt.rank == 0:
        self.logger.print_log(self.hook_info())

    def run(self):
        self.call_hook("before_run")
        while not self.finish:
            self.train()
        self.call_hook("after_run")

    @property
    def finish(self):
        if self.max_epoch:
            return self.epoch >= self.max_epoch
        else:
            return self.iter >= self.max_iter

    def train(self):
        self.model.train()
        self.call_hook("before_epoch")
        for batch_idx, (data) in enumerate(self.train_dataset):
            images = data['img']
            img_metas = data['img_metas']
            gt = data['gt_semantic_seg']
            self.outputs["batch_size"] = len(gt) * jt.world_size
            self.outputs["batch_idx"] = batch_idx
            self.outputs["data"] = data
            self.call_hook("before_iter")
            losses = self.model(img=images,
                                img_metas=img_metas,
                                gt_semantic_seg=gt)
            self.outputs["losses"] = losses
            self.call_hook("after_iter")
            self.iter += 1

            if self.finish:
                break
        self.call_hook("after_epoch")
        self.epoch += 1

    def register_train_hooks(self):
        self.register_hook(dict(type="LossParserHook", priority=Priority.VERY_HIGH))
        self.register_hook(dict(type="OptimizerHook", priority=Priority.HIGH))
        self.register_hook(dict(type="LRSchedulerHook", priority=Priority.ABOVE_NORMAL, by_epoch=False))
        self.register_hook(dict(type="LogPrinterHook", priority=Priority.NORMAL, log_interval=self.cfg.log_interval))
        self.register_hook(dict(type="CheckpointHook", priority=Priority.BELOW_NORMAL,
                                checkpoint_interval=self.cfg.checkpoint_interval))
        self.register_hook(dict(type="EvalHook", priority=Priority.LOW, eval_interval=self.cfg.eval_interval,
                                efficient_val=self.cfg.efficient_val))

    def register_custom_hooks(self):
        if self.cfg.custom_hooks is not None:
            assert isinstance(self.cfg.custom_hooks, list), "`custom_hooks` should be a list"
            for h in self.cfg.custom_hooks:
                self.register_hook(h)
