import argparse

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
        print("SPLIT!")
    elif args.command == 'benchmark':
        print("BENCHMARK!")
    elif args.command == 'capture':
        print("CAPTURE!")
    elif args.command == 'train':
        print("TRAIN!")
    else:
        print("xD")
