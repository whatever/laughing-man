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
