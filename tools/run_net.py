from utils.config_process import parser
import jittor as jt
from jittorseg.runner import TrainRunner, EvalRunner, TestRunner
from jittorseg.config import init_cfg
from jittorseg.config.config import update_cfg

jt.cudnn.set_max_workspace_ratio(0.0)


def main():
    args = parser()
    if not args.no_cuda:
        jt.flags.use_cuda = 1

    assert args.task in [
        "train", "val", "test"
    ], f"{args.task} not support, please choose [train,val,test]"

    if args.config_file:
        init_cfg(args.config_file)

    if args.resume:
        update_cfg(resume_path=args.resume)
    if args.efficient_val:
        update_cfg(efficient_val=args.efficient_val)

    runner = None

    if args.task == "train":
        runner = TrainRunner()
    elif args.task == "val":
        runner = EvalRunner()
    elif args.task == "test":
        runner = TestRunner(save_dir=args.save_dir)

    runner.run()


if __name__ == "__main__":
    main()
