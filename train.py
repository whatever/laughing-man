#!/usr/bin/env python3


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


class IsMattModule(torch.nn.Module):
    """..."""

    def __init__(self):
        super(IsMattModule, self).__init__()
        self.vgg16 = torchvision.models.vgg16(pretrained=True)

        for p in self.vgg16.parameters():
            p.requires_grad = False

        # XXX: UNTESTED!
        self.f1 = torch.nn.Sequential(
            torch.nn.MaxPool2d(1000, stride=1),
            torch.nn.Linear(2048, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 1),
            torch.nn.Sigmoid(),
        )

        # XXX: UNTESTED!
        self.f2 =  torch.nn.Sequential(
            torch.nn.MaxPool2d(1000, stride=1),
            torch.nn.Linear(2048, 2048),
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
    arr = transform(img.convert("RGB"))
    arr = torch.unsqueeze(arr, 0).to("cuda")
    return arr


def show_image(img):
    plot.figure()
    plot.imshow(F.to_pil_image(img.to("cpu")))
    plot.show()


if __name__ == "__main__":





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
