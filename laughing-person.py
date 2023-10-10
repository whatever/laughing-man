#!/usr/bin/env python3


import argparse
import cv2
import logging
import numpy as np
import torch
import signal
import sys

from datetime import timedelta
from datetime import datetime
from PIL import Image

from laughing_person import LaughingPerson


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def cleanup(dxD, cap):
    print("Leaving happily...")
    dxD.living = False
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--capture", type=int)
    args = parser.parse_args()

    print("xD")
    print("1.0.2")

    if args.checkpoint is None:
        logger.error("Checkpoint not specified... aborting")
        sys.exit(1)

    if args.capture is None:
        logger.error("No capture device specified... aborting")
        sys.exit(1)

    cap = cv2.VideoCapture(args.capture)

    if not cap.isOpened():
        logger.error("Cannot open capture device... aborting")
        sys.exit(1)

    xD = LaughingPerson(cap)

    signal.signal(signal.SIGINT, lambda sig, frame: cleanup(xD, cap))

    while xD.alive():
        ret, frame = xD.read()

        cv2.imshow("dxD", frame)

        if cv2.waitKey(1) == ord("q"):
            break

    cleanup(xD, cap)
