import argparse

import xD.benchmark
import xD.capture
import xD.split


def main():

    parser = argparse.ArgumentParser()
    
    subparsers = parser.add_subparsers(dest='command')

    split_parser = subparsers.add_parser('split')
    split_parser.add_argument("--num-augmentations", type=int, default=10)
    split_parser.add_argument("-v", "--verbose", action="store_true")
    
    benchmark_parser = subparsers.add_parser('benchmark')
    benchmark_parser.add_argument("--checkpoints", nargs="+", type=str)
    benchmark_parser.add_argument("--validate-dir", type=str, required=True)

    capture_parser = subparsers.add_parser('capture')

    train_parser = subparsers.add_parser('train')
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--checkpoint", type=str, required=True)
    train_parser.add_argument("--display", action="store_true")

    args = parser.parse_args()

    if args.command == 'split':
        xD.split.main(args.num_augmentations, args.verbose)

    elif args.command == 'benchmark':
        xD.benchmark.main(args.checkpoints, args.validate_dir)

    elif args.command == 'capture':
        xD.capture.main()

    elif args.command == 'train':
        xD.train.main(args.epochs, args.checkpoint, args.display)

    else:
        print("xD")
