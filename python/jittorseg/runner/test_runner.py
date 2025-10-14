import jittor as jt
from .base_runner import BaseRunner
from tqdm import tqdm
from jittorseg.utils.registry import build_from_cfg, DATASETS
from jittorseg.utils.visualize import visualize_result


class TestRunner(BaseRunner):

    def __init__(self, save_dir):
        super().__init__()
        self.test_dataset = build_from_cfg(self.cfg.dataset.test, DATASETS)
        self.save_dir = save_dir

    @jt.no_grad()
    @jt.single_process_scope()
    def test(self):
        if self.test_dataset is None:
            self.logger.print_log("Please set Test dataset")
        else:
            self.logger.print_log("Testing...")
            self.model.eval()
            for _, (data) in tqdm(enumerate(self.test_dataset)):
                images = data['img']
                img_metas = data['img_metas']
                results = self.model(images, img_metas, return_loss=False)
                visualize_result(results[0],
                                 palette=self.test_dataset.PALETTE,
                                 save_dir=self.save_dir,
                                 file_name=img_metas[0]['ori_filename'])

    def run(self):
        self.test()

