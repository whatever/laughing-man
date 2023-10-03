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
    alb.RandomCrop(width=1000, height=1000),
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


def convert_coords():
    pass


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

        coords = [
            min(p0[0], p1[0]),
            min(p0[1], p1[1]),
            max(p0[0], p1[0]),
            max(p0[1], p1[1]),
        ]

        int_coords = [int(p) for p in coords]

        image_fname = fname.replace("labels", "images").replace(".json", ".jpg")

        img = Image.open(image_fname).convert("RGB")

        w, h = img.size

        # Bounding Box Coords (normalized)
        bb = [
            coords[0] / w,
            coords[1] / h,
            coords[2] / w,
            coords[3] / h,
        ]

        # Preview and debug log

        print("image fname.......", image_fname)
        print("coords............", coords)
        print("width x height....", w, h)
        print("bounding box......", bb)


        augmented_image = augmentor(
            image=np.array(img),
            bboxes=[bb],
            class_labels=["matt-face"],
        )


        print(augmented_image)

        show_image(augmented_image["image"])

        # frame = cv2.imread(image_fname)
        # cv2.rectangle(frame, int_coords[0:2], int_coords[2:4], (0, 255, 0), 2)
        # cv2.imshow("Image", frame)
        # cv2.waitKey(0)

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
