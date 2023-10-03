#!/usr/bin/env python3


import albumentations as alb
import cv2
import matplotlib.pyplot as plot
import numpy as np
import json
import numpy
import torch
import torchvision

import torchvision.transforms.functional as F

from torchvision import transforms

from PIL import Image

torch.set_default_device('cuda')

from glob import glob

import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')


transform = transforms.Compose([
    transforms.Resize(size=256),
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ALBUMENTATION

ts = [
    alb.RandomCrop(width=450, height=450),
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


def load_image(path):
    """Return a PIL image from  apath"""
    img = Image.open(path)
    return img.convert("RGB")
    # arr = transform(img.convert("RGB"))
    # arr = torch.unsqueeze(arr, 0).to("cuda")
    return arr


def show_image(img):
    plot.figure()
    plot.imshow(img)
    # plot.imshow(F.to_pil_image(img.to("cpu")))
    plot.show()


if __name__ == "__main__":

    # fname = "/home/matt/git/whatever/laughing-person/data/test/images/img-1696267450-4.jpg"
    # frame  = cv2.imread(fname)
    # print(frame.shape)



    for fname in glob("data/test/labels/*.json"):
        with open(fname, "r") as fi:
            label = json.load(fi)

        shapes = label["shapes"]
        assert len(shapes) == 1


        shape = shapes[0]

        p0 = shape["points"][0]
        p1 = shape["points"][1]

        x0 = min(p0[0], p1[0])
        x1 = max(p0[0], p1[0])
        y0 = min(p0[1], p1[1])
        y1 = max(p0[1], p1[1])

        image_fname = fname.replace("labels", "images").replace(".json", ".jpg")
        frame = cv2.imread(image_fname)
        cv2.imshow("Image", frame)
        cv2.waitKey(0)


        raise "Start here... see what happens with drawing bounding box"


        # img = Image.open(image_fname).convert("RGB")
        # show_image(img)

    cv2.destroyAllWindows()




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
