import argparse

import xD
import xD.benchmark
import xD.capture
import xD.split
import xD.train


def main():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="command")

    parser.add_argument("-c", "--checkpoint", type=str, help="path to checkpoint file")
    parser.add_argument("-d", "--device", type=int, default=0, help="device number to use")
    parser.add_argument("-s", "--save-dir", type=str, default=".", help="path to directory to save image captures")

    benchmark_parser = subparsers.add_parser("benchmark")
    benchmark_parser.add_argument("--checkpoints", nargs="+", type=str)
    benchmark_parser.add_argument("--images-dir", type=str, required=True)
    benchmark_parser.add_argument("--labels-dir", type=str, required=True)
    benchmark_parser.add_argument("--sample-num", type=int, default=5)

    capture_parser = subparsers.add_parser("capture")
    capture_parser.add_argument("-d", "--device", type=int, default=0)

    split_parser = subparsers.add_parser("split")
    split_parser.add_argument("--num-augmentations", type=int, default=33)
    
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--epochs", type=int, default=10, help="number of epochs to run for training")
    train_parser.add_argument("-c", "--checkpoint", type=str, required=True, help="path to a pytorch checkpoint")
    train_parser.add_argument("-d", "--display", action="store_true", help="whether to display some examples of image classifications from the test set")

    args = parser.parse_args()


    print("xD")
    print("1.0.3")
    
    if args.command == "benchmark":
        xD.benchmark.main(args.checkpoints, args.images_dir, args.labels_dir)
    elif args.command == "capture":
        xD.capture.main(args.device)
    elif args.command == "split":
        xD.split.main(args.num_augmentations)
    elif args.command == "train":
        xD.train.main(args.epochs, args.checkpoint, args.display)
    else:
        render(args.checkpoint, args.device, args.save_dir)
