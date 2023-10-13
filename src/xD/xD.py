#!/usr/bin/env python3


import argparse


import cv2
import logging
import numpy as np
import os.path
import torch
import signal
import sys

from datetime import timedelta
from datetime import datetime
from PIL import Image
from xD.app import LaughingPerson


logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


import xD.benchmark
import xD.capture
import xD.split
import xD.train


def cleanup(dxD, cap):
    print("Leaving happily...")
    cap.release()
    cv2.destroyAllWindows()


save_map = {
    ord("p"): "tp",
    ord("f"): "fp",
    ord("n"): "fn",
}



def render(checkpoint, capture, save_dir):

    print()
    print("""
    p - true positive
    f - false positive
    n - false negative
    """)

    if checkpoint is None:
        logger.error("Checkpoint not specified... aborting")
        sys.exit(1)

    if capture is None:
        logger.error("No capture device specified... aborting")
        sys.exit(1)

    cap = cv2.VideoCapture(capture)

    if not cap.isOpened():
        logger.error("Cannot open capture device... aborting")
        sys.exit(1)

    xD = LaughingPerson(cap, checkpoint)

    signal.signal(signal.SIGINT, lambda sig, frame: cleanup(xD, cap))

    while xD.alive():
        ret, frame = xD.read()

        cv2.imshow("dxD", frame)

        c = cv2.waitKey(1)

        if c in save_map:
            label = save_map[c]
            now = int(datetime.now().timestamp())
            fname = os.path.join(
                save_dir,
                f"dxD-{now}-{label}.jpg"
            )
            logger.info(f"Saving frame to \"{fname}\"")
            cv2.imwrite(fname, xD.last_read[1])
        elif c == ord("q"):
            logger.info("Quitting...")
            break

    cleanup(xD, cap)


def main():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="command")

    parser.add_argument("-c", "--checkpoint", type=str, help="path to checkpoint file")
    parser.add_argument("-d", "--device", type=int, default=0, help="device number to use")
    parser.add_argument("-s", "--save-dir", type=str, default=".", help="path to directory to save image captures")

    benchmark_parser = subparsers.add_parser("benchmark")
    benchmark_parser.add_argument("--checkpoints", nargs="+", type=str)
    benchmark_parser.add_argument("--validate-dir", type=str, required=True)

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
        xD.benchmark.main(args.checkpoints, args.validate_dir)
    elif args.command == "capture":
        xD.capture.main(args.device)
    elif args.command == "split":
        xD.split.main(args.num_augmentations)
    elif args.command == "train":
        xD.train.main(args.epochs, args.checkpoint, args.display)
    else:
        render(args.checkpoint, args.device, args.save_dir)
