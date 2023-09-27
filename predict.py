#!/usr/bin/env python3


import json
import numpy
import torch
import torchvision

from torchvision import transforms

from PIL import Image

torch.set_default_device('cuda')

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

    def forward(self, x):
        x = self.vgg16(x)
        return x


if __name__ == "__main__":

    img = Image.open("heart.png")
    arr = transform(img.convert("RGB"))
    arr = torch.unsqueeze(arr, 0).to("cuda")

    with open("imagenet_class_index.json", "r") as fi:
        labels = json.load(fi)

    labels = {
        int(k): v[-1]
        for k, v in labels.items()
    }

    model = IsMattModule()

    for i in range(300):
        probs = model(arr)
        idx = torch.argmax(probs)
