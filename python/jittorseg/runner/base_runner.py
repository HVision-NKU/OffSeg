import jittor as jt
import jittorseg
from jittorseg.config import get_cfg, save_cfg, print_cfg
from jittorseg.utils.registry import build_from_cfg, MODELS, SCHEDULERS, HOOKS, OPTIMS
from jittorseg.utils.general import build_file, current_time, check_file, search_ckpt, is_method_overridden
from jittorseg.hooks import Hook

from abc import ABCMeta, abstractmethod


class BaseRunner(metaclass=ABCMeta):
    def __init__(self):
        cfg = get_cfg()
        self.cfg = cfg
        # self.flip_test = [] if cfg.flip_test is None else cfg.flip_test
        self.work_dir = cfg.work_dir

        self.max_epoch = cfg.max_epoch
        self.max_iter = cfg.max_iter
        assert (self.max_iter is None) ^ (
                self.max_epoch is None), "You must set max_iter or max_epoch"
        self.iter = 0
        self.epoch = 0

        self.resume_path = cfg.resume_path
        self.efficient_val = cfg.efficient_val

        self.logger = build_from_cfg(self.cfg.logger,
                                     HOOKS,
                                     work_dir=self.work_dir)

        self.model = build_from_cfg(cfg.model, MODELS)
        if jt.rank == 0:
            print_cfg()
            self.logger.log({'model': self.model})

        if (cfg.parameter_groups_generator):
            params = build_from_cfg(cfg.parameter_groups_generator,
                                    MODELS,
                                    named_params=self.model.named_parameters(),
                                    model=self.model,
                                    logger=self.logger)
        else:
            params = self.model.parameters()
        self.optimizer = build_from_cfg(cfg.optimizer, OPTIMS, params=params)
        self.scheduler = build_from_cfg(cfg.scheduler,
                                        SCHEDULERS,
                                        optimizer=self.optimizer)
        self.hooks = []
        self.outputs = {}

        save_file = build_file(self.work_dir, prefix="config.yaml")
        save_cfg(save_file)

        if (cfg.pretrained_weights):
            self.load(cfg.pretrained_weights, model_only=True)

        # TODO: Resuming is auto-executed now. If the `pretrained_weights` is set, the weights from pretrained model
        #  will be overwrited by the resumed one.

        if self.resume_path is None:
            self.resume_path = search_ckpt(self.work_dir)
        if check_file(self.resume_path):
            self.resume()

    @abstractmethod
    def run(self, *args, **kwargs):
        pass

    # def test_time(self):
    #     warmup = 10
    #     rerun = 100
    #     self.model.train()
    #     for batch_idx, (data) in enumerate(self.train_dataset):
    #         break
    #     images = data['img']
    #     img_metas = data['img_metas']
    #     gt = data['gt_semantic_seg']
    #     print("warmup...")
    #     for i in tqdm(range(warmup)):
    #         losses = self.model(img=images,
    #                             img_metas=img_metas,
    #                             gt_semantic_seg=gt)
    #         all_loss, losses = parse_losses(losses)
    #         self.optimizer.step(all_loss)
    #         self.scheduler.step(self.iter, self.epoch, by_epoch=True)
    #     jt.sync_all(True)
    #     print("testing...")
    #     start_time = time.time()
    #     for i in tqdm(range(rerun)):
    #         losses = self.model(img=images,
    #                             img_metas=img_metas,
    #                             gt_semantic_seg=gt)
    #         all_loss, losses = parse_losses(losses)
    #         self.optimizer.step(all_loss)
    #         self.scheduler.step(self.iter, self.epoch, by_epoch=True)
    #     jt.sync_all(True)
    #     batch_size = len(gt) * jt.world_size
    #     ptime = time.time() - start_time
    #     fps = batch_size * rerun / ptime
    #     print("FPS:", fps)

    @jt.single_process_scope()
    def save(self):
        save_data = {
            "meta": {
                "jittorseg_version": jittorseg.__version__,
                "epoch": self.epoch,
                "iter": self.iter + 1,
                "max_iter": self.max_iter,
                "max_epoch": self.max_epoch,
                "save_time": current_time(),
                "config": self.cfg.dump()
            },
            "model": self.model.state_dict(),
            "scheduler": self.scheduler.parameters(),
            "optimizer": self.optimizer.parameters()
        }

        save_file = build_file(self.work_dir,
                               prefix=f"checkpoints/ckpt_{self.iter}.pkl")
        jt.save(save_data, save_file)
        print("saved")

    def load(self, load_path, model_only=False):
        resume_data = jt.load(load_path)

        if not model_only:
            meta = resume_data.get("meta", dict())
            self.epoch = meta.get("epoch", self.epoch)
            self.iter = meta.get("iter", self.iter)
            self.max_iter = meta.get("max_iter", self.max_iter)
            self.max_epoch = meta.get("max_epoch", self.max_epoch)
            self.scheduler.load_parameters(resume_data.get(
                "scheduler", dict()))
            self.optimizer.load_parameters(resume_data.get(
                "optimizer", dict()))
        if "model" in resume_data:
            state_dict = resume_data["model"]
        elif "state_dict" in resume_data:
            state_dict = resume_data["state_dict"]
        else:
            state_dict = resume_data
        self.model.load_parameters(state_dict)
        self.logger.print_log(
            f"Missing key: {self.model.state_dict().keys() - state_dict.keys()}"
        )
        self.logger.print_log(f"Loading model parameters from {load_path}")

    def resume(self):
        self.load(self.resume_path)

    def register_hook(self, hook_cfg):
        hook = build_from_cfg(hook_cfg, HOOKS, runner=self)
        idx = 0
        ok = False
        for h in self.hooks:
            if hook.priority < h.priority:
                ok = True
                break
            idx += 1
        if ok:
            self.hooks.insert(idx, hook)
        else:
            self.hooks.append(hook)

    def call_hook(self, hook_stage):
        for h in self.hooks:
            getattr(h, hook_stage)()

    def hook_info(self):
        info = {}
        for stage in Hook.stages:
            info[stage] = []
            for h in self.hooks:
                if is_method_overridden(stage, Hook, h):
                    info[stage].append((h.name, h.priority))

        info_str = "\n"
        for k, v in info.items():
            info_str += f"Stage {k}:\n"
            info_str += f"{'Name':^20}|{'Prio':^10}\n"
            info_str += '-' * 30 + '\n'
            for pair in v:
                info_str += f"{pair[0]:^20}|{pair[1]:^10}\n"
            info_str += '-' * 30 + '\n'
        return info_str








