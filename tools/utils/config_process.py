import argparse


def parser():
    parser = argparse.ArgumentParser(
        description="OpenJittorVision Semantic segmentation Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--task",
        default="train",
        help="train,val test",
        type=str,
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="resume path",
        type=str,
    )
    parser.add_argument(
        "--save-dir",
        default="./results",
        type=str,
    )
    parser.add_argument("--no_cuda", action='store_true')
    parser.add_argument("--efficient_val", action='store_true')
    return parser.parse_args()
