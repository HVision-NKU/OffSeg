import jittor as jt
from jittorseg.utils.registry import HOOKS
from .hook import Hook
from jittorseg.utils.general import check_interval
from tqdm import tqdm
from jittorseg.utils.general import np2tmp


@HOOKS.register_module()
class EvalHook(Hook):
    def __init__(self, runner, priority, eval_interval, efficient_val):
        super(EvalHook, self).__init__(runner=runner, priority=priority)
        self.eval_interval = eval_interval
        self.efficient_val = efficient_val

    def after_iter(self):
        if check_interval(self.runner.iter + 1, self.eval_interval):
            self.val()
            self.runner.model.train()

    @jt.no_grad()
    @jt.single_process_scope()
    def val(self):
        if self.runner.val_dataset is None:
            self.runner.logger.print_log("Please set Val dataset")
        else:
            self.runner.logger.print_log("Validating....")
            if self.runner.model.is_training():
                self.runner.model.eval()
            results = []
            for _, (data) in tqdm(enumerate(self.runner.val_dataset)):
                images = data['img']
                img_metas = data['img_metas']
                result = self.runner.model(images, img_metas, return_loss=False)
                jt.sync_all(True)
                if isinstance(result, list):
                    if self.efficient_val:
                        result = [np2tmp(_) for _ in result]
                    results.extend(result)
                else:
                    if self.efficient_val:
                        result = np2tmp(result)
                    results.append(result)

            self.runner.val_dataset.evaluate(results,
                                             metric='mIoU',
                                             logger=self.runner.logger)
