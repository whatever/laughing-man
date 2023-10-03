#!/usr/bin/env python3


import albumentations as alb
import cv2
import logging
import matplotlib.pyplot as plot
import numpy as np
import json
import numpy
import os.path
import random
import torch
import torchvision

import torchvision.transforms.functional as F

from collections import defaultdict

from torchvision import transforms

from PIL import Image

torch.set_default_device('cuda')

from glob import glob

import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')


tensorify = transforms.ToTensor()


transform = transforms.Compose([
    transforms.Resize(size=256),
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


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


class IsMattModule(torch.nn.Module):
    """..."""

    def __init__(self):
        super(IsMattModule, self).__init__()
        self.vgg16 = torchvision.models.vgg16(pretrained=True)

        for p in self.vgg16.parameters():
            p.requires_grad = False

        self.face = torch.nn.Sequential(
            torch.nn.AdaptiveMaxPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(512, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 1),
            torch.nn.Sigmoid(),
        )

        self.loc =  torch.nn.Sequential(
            torch.nn.AdaptiveMaxPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(512, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 4),
            torch.nn.Sigmoid(),
        )


    def forward(self, x):
        x = self.vgg16(x)
        return x


def load_image_old(path):
    """Return a PIL image from  apath"""
    img = Image.open(path)
    return img.convert("RGB")
    # arr = transform(img.convert("RGB"))
    # arr = torch.unsqueeze(arr, 0).to("cuda")
    return arr


def show_image(img, bboxes=list()):
    """Show a channels first image"""

    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    w, h, _ = frame.shape

    if len(bboxes) == 0:
        print("No bounding box")

    for bb in bboxes:
        print("Bounding Box:", bb)
        cv2.rectangle(
            frame,
            (int(bb[0]*w), int(bb[1]*h)),
            (int(bb[2]*w), int(bb[3]*h)),
            (255, 0, 0),
            2,
        )

    cv2.imshow("image", frame)


def get_label_fname(image_fname):
    return (
        image_fname
        .replace("images", "labels")
        .replace(".jpg", ".json")
    )


def load_image(image_path, f=transform):
    with Image.open(image_path) as img:
        return img.convert("RGB")


def load_label(label_path):
    with open (label_path, "r") as fi:
        label = json.load(fi)
    return np.array([label["class"]]).astype(np.uint8), np.array(label["bbox"]).astype(np.float16)


def dataset(partition):
    """Yield (image, bounding box) from a partition of the dataset"""

    files = list(glob(f"aug_data/{partition}/images/*.jpg"))

    random.shuffle(files)

    for image_fname in files:
        label_fname = get_label_fname(image_fname)

        if not os.path.exists(image_fname):
            logging.warn("Missing image:", image_fname, "... skipping")
            continue

        if not os.path.exists(label_fname):
            logging.warn("Missing image:", label_fname, "... skipping")
            continue

        yield load_image(image_fname), load_label(label_fname)



if __name__ == "__main__":


    images = defaultdict(list)

    labels = defaultdict(list)

    print("Print showing 5 images to sanity-check")

    sample = dataset("train")
    sample = [next(sample) for _ in range(5)]

    for img, bb in sample:
        show_image(img, bb[1])
        print(bb)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    raise SystemExit

    print(type(img))
    print(type(img))

    show_image(img)


    img2 = transform(load_image_old("heart.png"))
    print(type(img2))
    print(img2.shape)



    raise SystemExit
    

    for partition in ["train", "test", "validate"]:
        for label_fname in glob(f"data/{partition}/labels/*.json"):
            image_fname = label_fname.replace("labels", "images").replace(".json", ".jpg")
            aug_fname = os.path.join("aug_data", partition, "labels", os.path.basename(label_fname))



            if not os.path.exists(image_fname):
                annotation["bbox"] = [0]*4
                annotation["class"] = 0

            with open(aug_fname, "w") as f:
                # json.dump(aug_
                pass

            print(label_fname)
            print(image_fname)
            print(aug_fname)




    raise SystemExit

    with open("imagenet_class_index.json", "r") as fi:
        labels =  {
            int(k): v[-1]
            for k, v in json.load(fi).items()
        }

    img1 = torchvision.io.read_image("heart.png")
    img1 = img1[None, :, :, :]

    arr = load_image("heart.png")
    show_image(arr[0])
    model = IsMattModule()
    probs = model(arr)
    idx = torch.argmax(probs)

    print("Image is:", labels[int(idx)])

    raise SystemExit

    img1 = torchvision.io.read_image("heart.png")
    img1 = img1[None, :, :, :]

    print(img1.shape)

    img2 = load_image("heart.png")

    print(img2.shape)

    show_image(img1[0])

    raise SystemExit

    fnames = [
        fname
        for fname in glob("imgs-annotated/*.json")
    ]

    images = [
        load_image(fname)
        for fname in glob("imgs/*.jpg")
    ]

    print(images)


    raise SystemExit

    for fname in glob("imgs/*.jpg"):
        print(fname)



    raise SystemExit

    arr = load_image("heart.png")


    with open("imagenet_class_index.json", "r") as fi:
        labels =  {
            int(k): v[-1]
            for k, v in json.load(fi).items()
        }

    model = IsMattModule()
    probs = model(arr)
    idx = torch.argmax(probs)
