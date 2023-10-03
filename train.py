#!/usr/bin/env python3


import albumentations as alb
import argparse
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

import torch.nn.functional as F

from collections import defaultdict

from torchvision import transforms
import torchvision.models.vgg as vgg

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

        self.vgg16 = torchvision.models.vgg16(weights=vgg.VGG16_Weights.DEFAULT).to("cuda")

        for p in self.vgg16.parameters():
            p.requires_grad = False

        self.glob = torch.nn.AdaptiveMaxPool2d(1)

        self.face = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(512, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 1),
            torch.nn.Sigmoid(),
        )

        self.loc = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(512, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 4),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.vgg16.features(x)
        print(x.shape)

        loc = F.max_pool2d(x, kernel_size=7)
        fac = F.max_pool2d(x, kernel_size=7)

        print(self.glob(x).shape)
        print(loc.shape)
        print(fac.shape)

        return self.face(loc), self.loc(fac)


def show_image(img, bboxes=list()):
    """Show a channels first image"""


    for i in range(img.shape[0]):
        arr = img[i].cpu().numpy()
        arr = np.moveaxis(arr, 0, 2)
        frame = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

        w, h, _ = frame.shape

        for bb in bboxes:
            cv2.rectangle(
                frame,
                (int(bb[0]*w), int(bb[1]*h)),
                (int(bb[2]*w), int(bb[3]*h)),
                (0, 255, 255),
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
    with Image.open(image_path) as arr:
        arr = arr.convert("RGB")
        arr = transform(arr)
        arr = torch.unsqueeze(arr, 0)
        arr = arr.cuda()
        return arr


def load_label(label_path):
    with open (label_path, "r") as fi:
        label = json.load(fi)
    bbox = label["bbox"]
    bbox = bbox if bbox else [[0.0]*4]
    return (
        torch.tensor([[label["class"]]], dtype=torch.uint8),
        torch.tensor(bbox, dtype=torch.float32),
    )


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


with open("imagenet_class_index.json", "r") as fi:
    LABELS =  {
        int(k): v[-1]
        for k, v in json.load(fi).items()
    }


def predict(arr):
    probs = model(arr)
    return probs
    idx = torch.argmax(probs)
    return LABELS[int(idx)]


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--display", action="store_true")
    args = parser.parse_args()

    model = IsMattModule()

    optim = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])
        optim.load_state_dict(checkpoint["optim_state_dict"])
        epoch = checkpoint["epoch"]
        loss = checkpoint["loss"]




        print("+===========================+")
        print("| epoch ..............", epoch)
        print("| batch loss .........", loss)
        print("+===========================+")
        print()
        print()
        print()

    # Samples

    sample = dataset("train")
    sample = [next(sample) for _ in range(5)]

    """
    for img, bb in sample:
        show_image(img, bb[1])
        cv2.waitKey(0)
        print("Predict =", predict(img))
    else:
        cv2.destroyAllWindows()
    """


    for epoch in range(args.epochs):

        print("+===========================+")
        print("| epoch ..............", epoch)

        optim.zero_grad()

        i = 0
        last_loss = 0.0
        last_loca_loss = 0.0
        last_face_loss = 0.0

        for img, bbox in dataset("train"):
            y_hat_face, y_hat_loca = model(img)

            loca_loss = torch.nn.functional.mse_loss(y_hat_loca, bbox[1])
            face_loss = torch.nn.functional.binary_cross_entropy(y_hat_face, bbox[0].float())
            loss = loca_loss + 0.5*face_loss

            loss.backward()

            optim.step()

            last_loca_loss += loca_loss
            last_face_loss += face_loss
            last_loss += loss
            i += 1


        print("| batch loca loss ....", last_loca_loss)
        print("| batch face loss ....", last_face_loss)
        print("| batch loss .........", last_loss/i)
        print("+===========================+")
        print()


        samp = dataset("validate")
        samp = [next(samp) for _ in range(5)]

        for img, bb in samp:
            y_hat_face, y_hat_loca = model(img)
            print(y_hat_face, bb[0])
            print(y_hat_loca, bb[1])




    if args.display:
        samp = dataset("validate")
        samp = [next(samp) for _ in range(5)]
        for img, bb in samp:
            y_hat_face, y_hat_loca = model(img)
            print(y_hat_face)
            print(y_hat_loca)
            show_image(img, bb[1])
            cv2.waitKey(0)

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optim_state_dict": optim.state_dict(),
        "loss": last_loss,
    }, args.checkpoint)