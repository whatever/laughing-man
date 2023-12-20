#!/usr/bin/env python3


import json
import numpy as np
import os.path
import torch
import xD

from PIL import Image
from glob import glob
from torch.utils.data import Dataset, DataLoader


class Dataset(Dataset):
    """Dataset"""

    def __init__(self, images_dir, labels_dir, verbose=False):
        """Construct a dataset from matching filenames between an images and labels"""
        self.image_filenames = glob(os.path.join(images_dir, "*.jpg"))
        self.label_dir = os.path.dirname(labels_dir)
        self.label_filenames = {
            os.path.basename(f): f
            for f in glob(os.path.join(labels_dir, "*.json"))
        }

    @staticmethod
    def _load_image(image_path):
        with Image.open(image_path) as img:
            img = img.copy()
            img = img.convert("RGB")
            img = xD.crop(img)

            print(type(img))

            arr = xD.transform(img)
            arr = arr.to(xD.DEVICE)
        return img, arr

    def __len__(self):
        """Return number of images that we're doing"""
        return len(self.image_filenames)

    def __getitem__(self, idx):
        """Return """
        img_fname = self.image_filenames[idx]
        lab_fname = os.path.basename(img_fname).replace("jpg", "json")
        lab_fname = os.path.join(self.label_dir, lab_fname)

        img, arr = Dataset._load_image(img_fname)

        print("3", type(img))

        if not os.path.exists(lab_fname):
            label = {}
        else:
            with open(lab_fname, "r") as f:
                label = json.load(f)
                if len(label["bbox"]) == 1:
                    label["bbox"] = label["bbox"][0]

        return {
            "idx": idx,
            "image": img,
            "fname": img_fname,
            "image": arr,
            "face": torch.tensor([label.get("class", 0)]),
            "bbox": torch.tensor(label.get("bbox", [0., 0., 0.0001, 0.0001])),
        }
