#!/usr/bin/env python3


import argparse
import cv2
import laughing_person as lp
import logging
import json
import numpy as np
import torch
import signal


from datetime import timedelta
from datetime import datetime
from glob import glob
from train import dataset, IsMattModule, get_label_fname, load_label
from PIL import Image

face_cascade = cv2.CascadeClassifier('capture-images/haarcascade_frontalface_default.xml')


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def cleanup(signum, frame):
    raise SystemExit


def fix_bb(bb):
    return [
        min(bb[0], bb[2]),
        min(bb[1], bb[3]),
        max(bb[0], bb[2]),
        max(bb[1], bb[3]),
    ]

def calc_iou(bb1, bb2):
    bb1 = fix_bb(bb1)
    bb2 = fix_bb(bb2)

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[2], bb2[2])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0]) * (bb1[2] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[2] - bb2[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-dir", type=str, required=True)
    parser.add_argument("--checkpoints-dir", type=str, required=True)
    args = parser.parse_args()

    model = IsMattModule()

    for model_fname in sorted(glob(f"{args.checkpoints_dir}/*.pt")):
        logger.info("Loading model %s", model_fname)
        checkpoint = torch.load(model_fname)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model = model.cuda()

        logging.info("Validating model %s", model_fname)

        examples = sorted(fname for fname in glob("data/train/images/*.jpg"))
        examples = examples[0:1]


        for image_fname in examples:
            label_fname = get_label_fname(image_fname)
            with open(label_fname, "r") as fi:
                label = json.load(fi)

            is_face_actual = len(label["shapes"])

            img = Image.open(image_fname)
            face, bb = model.predict(img)
            p, q = label["shapes"][0]["points"]
            predicted_bb = p + q


            w, h = img.size


            y = predicted_bb
            y_hat = (bb[0].cpu().numpy()*np.array([w, h, w, h])).tolist()
            print(image_fname, calc_iou(y, y_hat))
