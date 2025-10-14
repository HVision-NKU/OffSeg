import jittor as jt
from jittorseg.utils.registry import HOOKS
from .hook import Hook
from jittorseg.utils.general import check_interval, sync
import time
import datetime

@HOOKS.register_module()
class LogPrinterHook(Hook):
    def __init__(self, runner, priority, log_interval):
        super(LogPrinterHook, self).__init__(runner=runner, priority=priority)
        self.log_interval = log_interval

    def before_run(self):
        self.runner.logger.print_log("Begin running")

    def before_iter(self):
        self.runner.outputs["start_time"] = time.time()

    def after_iter(self):
        if check_interval(self.runner.iter + 1, self.log_interval):
            start_time = self.runner.outputs["start_time"]
            ptime = time.time() - start_time
            batch_idx = self.runner.outputs["batch_idx"]
            all_loss = self.runner.outputs["all_loss"]
            losses = self.runner.outputs["losses"]
            batch_size = self.runner.outputs["batch_size"]
            fps = batch_size * (batch_idx + 1) / ptime
            eta_time = (self.runner.total_iter - self.runner.iter - 1) * ptime / (batch_idx +
                                                                1)
            eta_str = str(datetime.timedelta(seconds=int(eta_time)))
            data = dict(name=self.runner.cfg.name,
                        lr=self.runner.optimizer.cur_lr(),
                        iter=self.runner.iter + 1,
                        epoch=self.runner.epoch,
                        batch_idx=batch_idx,
                        batch_size=batch_size,
                        total_loss=all_loss,
                        fps=fps,
                        eta=eta_str)
            data.update(losses)
            data = sync(data)
            # is_main use jt.rank==0, so it's scope must have no jt.Vars
            if jt.rank == 0:
                self.runner.logger.log(data)

    def after_run(self):
        self.runner.logger.print_log("After running")