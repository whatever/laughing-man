#!/usr/bin/env python3

import json
import numpy
import torch
import torch.nn.functional as F
import torchvision
import warnings

from collections import defaultdict
from glob import glob
from PIL import Image
from torchvision import transforms


torch.set_default_device('cuda')

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

        vgg16 = torchvision.models.vgg16(weights=torchvision.models.vgg.VGG16_Weights.DEFAULT)

        self.model = vgg16.features

        for p in self.model.parameters():
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
        x = self.model(x)
        return self.face(x), self.loc(x)


def load_image(path):
    """Return a PIL image from  apath"""
    img = Image.open(path)
    return img
    arr = transform(img.convert("RGB"))
    arr = torch.unsqueeze(arr, 0).to("cuda")
    return arr


if __name__ == "__main__":

    img = Image.open("heart.png")
    arr = transform(img.convert("RGB"))
    arr = torch.unsqueeze(arr, 0).to("cuda")

    with open("imagenet_class_index.json", "r") as fi:
        labels =  {
            int(k): v[-1]
            for k, v in json.load(fi).items()
        }

    model = IsMattModule()

    face, loc = model(arr)

    print(face.shape)
    print(loc.shape)
