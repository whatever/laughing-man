#!/usr/bin/env python3


import argparse
import os.path
import json
import numpy as np
import random
import shutil

import laughing_person as lp


from collections import defaultdict
from glob import glob
from PIL import Image


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--num-augmentations", type=int, default=10)
    args = parser.parse_args()

    # To make the the run immutable
    random.seed(420)

    positive_image_fnames = [
        fname
        for fname in glob("images/img-*-*.jpg")
    ]

    negative_image_fnames = random.sample([
        fname
        for fname in glob("images/image_*.jpg")
    ], len(positive_image_fnames))


    image_fnames = sorted(positive_image_fnames + negative_image_fnames)


    labels = []


    for image_fname in image_fnames:

        label_fname = (
            image_fname
            .replace("images", "labels")
            .replace(".jpg", ".json")
        )

        label = {
            "image_fname": image_fname,
            "label_fname": None,
            "class": 0,
            "bbox": [0., 0., 0.0000001, 0.0000001],
        }

        if os.path.exists(label_fname):

            with open(label_fname, "r") as f:
                lab = json.load(f)
                del lab["imageData"]

            img = Image.open(image_fname)
            w, h = img.size
            del img

            p, q = lab["shapes"][0]["points"]
            x0, y0, x1, y1 = p + q
            x0, x1 = sorted([x0, x1])
            y0, y1 = sorted([y0, y1])

            bbox = [
                x0 / w,
                y0 / h,
                x1 / w,
                y1 / h,
            ]

            label = {
                "image_fname": image_fname,
                "label_fname": label_fname,
                "class": 1,
                "bbox": bbox,
            }

        labels.append(label)


    """
    # SPLIT INTO TRAIN, TEST, VALIDATE

    n_train = int(0.7 * len(labels))
    n_test = int(0.15 * len(labels))
    n_validate = len(labels) - n_train - n_test

    random.shuffle(labels)

    split = {
        "train": labels[:n_train],
        "test": labels[n_train:(n_train + n_test)],
        "validate": labels[n_train+n_test:],
    }
    """


    summary = defaultdict(int)

    # AUGMENT and COPY

    random.shuffle(labels)

    for item in labels:

        img = Image.open(item["image_fname"]).convert("RGB")

        for i in range(33):
            res = lp.augmentor(
                image=np.array(img),
                bboxes=[item["bbox"]],
                class_labels=[item["class"]],
            )

            assert len(res["bboxes"]) < 2
            assert len(res["class_labels"]) < 2

            face = res["class_labels"][0] if len(res["class_labels"]) > 0 else 0
            bbox = res["bboxes"] or [0., 0., 0., 0.]


            fname = os.path.basename(item["image_fname"])
            fname, suffix = fname.split(".", 1)
            fname = f"{fname}--{i}.{suffix}"

            r = random.random()

            if r < 0.15:
                partition = "validate"
            elif r < 0.3:
                partition = "test"
            else:
                partition = "train"

            summary[partition] += 1

            aug_image_fname = os.path.join("augmented-data", partition, "images", fname)
            os.makedirs(os.path.dirname(aug_image_fname), exist_ok=True)
            Image.fromarray(res["image"]).save(aug_image_fname)

            aug_label_fname = aug_image_fname.replace("images", "labels").replace(".jpg", ".json")
            os.makedirs(os.path.dirname(aug_label_fname), exist_ok=True)

            with open(aug_label_fname, "w") as f:
                json.dump({
                    "bbox": bbox,
                    "class": face,
                    "image_fname": aug_image_fname,
                }, f)

            summary["face"] += face
            summary["total"] += 1
    

    print(json.dumps(summary, indent=4))
