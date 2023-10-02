#!/usr/bin/env python3


import json
import numpy
import torch
import torchvision

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
    return img
    arr = transform(img.convert("RGB"))
    arr = torch.unsqueeze(arr, 0).to("cuda")
    return arr


if __name__ == "__main__":

    for fname in glob("capture/imgs/*.jpg"):
        print(fname)

    with open("") as fi:
        json.load(fi)



    raise SystemExit

    arr = load_image("heart.png")


    with open("imagenet_class_index.json", "r") as fi:
        labels =  {
            int(k): v[-1]
            for k, v in json.load(fi).items()
        }

    model = IsMattModule()

    print(model)

    probs = model(arr)
    idx = torch.argmax(probs)
