#!/usr/bin/env python3


import albumentations as alb
import cv2
import matplotlib.pyplot as plot
import numpy as np
import json
import numpy
import os.path
import torch
import torchvision

import torchvision.transforms.functional as F

from torchvision import transforms

from PIL import Image

torch.set_default_device('cuda')

from glob import glob

import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')


# ALBUMENTATION

CROP_WIDTH = CROP_HEIGHT = 1000


ts = [
    alb.RandomCrop(width=CROP_WIDTH, height=CROP_HEIGHT),
    alb.HorizontalFlip(p=0.5),
    alb.VerticalFlip(p=0.5),
    alb.RandomBrightnessContrast(p=0.2),
    alb.RandomGamma(p=0.2),
    alb.RGBShift(p=0.2),
]

bbox_params = alb.BboxParams(
    format="albumentations",
    label_fields=["class_labels"],
)

augmentor = alb.Compose(ts, bbox_params)


def load_image(path):
    """Return a PIL image from  apath"""
    img = Image.open(path)
    return img.convert("RGB")


def show_augmented_image(aug):
    img = aug["image"].copy()
    for bb in aug["bboxes"]:
        cv2.rectangle(
            img,
            tuple(np.multiply(aug["bboxes"][0][0:2], CROP_WIDTH).astype(int)),
            tuple(np.multiply(aug["bboxes"][0][2:], CROP_HEIGHT).astype(int)),
            (0, 255, 0),
            2,
        )
    show_image(img)


def get_label_fname(image_fname):
    return (
        image_fname
        .replace("images", "labels")
        .replace(".jpg", ".json")
    )


def get_image_and_label(image_fname):

    label_fname = get_label_fname(image_fname)

    with open(label_fname, "r") as fi:
        label = json.load(fi)

    shape = label.get("shapes", [None])[0]


    img = Image.open(image_fname).convert("RGB")
    w, h = img.size


    if not shape:
        return {
            "image_fname": image_fname,
            "label_fname": label_fname,
            "bbox": [0, 0, 0.000001,  0.000001],
            "class": 0,
        }, img

    p0, p1 = shape["points"]

    coords = [
        min(p0[0], p1[0]),
        min(p0[1], p1[1]),
        max(p0[0], p1[0]),
        max(p0[1], p1[1]),
    ]

    bb = [
        coords[0] / w,
        coords[1] / h,
        coords[2] / w,
        coords[3] / h,
    ]

    class_label = int(shape["label"] == "matt-face")

    res = {
        "image_fname": image_fname,
        "label_fname": label_fname,
        "bbox": bb,
        "class": class_label,
    }

    return res, img



if __name__ == "__main__":
    for partition in ["train", "test", "validate"]:
        for image_fname in glob(f"data/{partition}/images/*.jpg"):
            res, img = get_image_and_label(image_fname)

            for i in range(60):
                aug = augmentor(
                    image=np.array(img),
                    bboxes=[res["bbox"]],
                    class_labels=["face"],
                )

                aug_basename = "{}-{}".format(
                    os.path.basename(image_fname).replace(".jpg", ""),
                    i,
                )

                aug_image_fname = os.path.join(
                    "aug_data",
                    partition,
                    "images",
                    f"{aug_basename}.jpg",
                )

                aug_label_fname = os.path.join(
                    "aug_data",
                    partition,
                    "labels",
                    f"{aug_basename}.json",
                )


                assert(len(aug["class_labels"]) < 2)

                annotation = {
                    "bbox": aug.get("bboxes", [0., 0., 0., 0.]),
                    "class": len(aug["class_labels"]),
                    "image": f"{aug_basename}.jpg",
                }

                aug_img = Image.fromarray(aug["image"])
                aug_img.save(aug_image_fname)

                with open(aug_label_fname, "w") as f:
                    json.dump(annotation, f)
