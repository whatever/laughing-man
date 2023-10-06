#!/usr/bin/env python3


import albumentations as alb
import argparse
import cv2
import laughing_person as lp
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

from datetime import datetime
from PIL import Image

torch.set_default_device('cuda')

from glob import glob

import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')


class IsMattModule(torch.nn.Module):
    """..."""

    def __init__(self, freeze_vgg=True):

        super(IsMattModule, self).__init__()

        self.vgg16 = torchvision.models.vgg16(weights=vgg.VGG16_Weights.DEFAULT).to("cuda")

        for p in self.vgg16.parameters():
            p.requires_grad = freeze_vgg

        self.face = torch.nn.Sequential(
            torch.nn.MaxPool2d(7),
            torch.nn.Flatten(),
            torch.nn.Linear(512, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 1),
            torch.nn.Sigmoid(),
        )

        self.loc = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(7*7*512, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 4),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.vgg16.features(x)
        return self.face(x), self.loc(x)

    def predict(self, img):
        with torch.no_grad():
            img = lp.transform(img)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()
            probs = self.forward(img)
            return probs



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


def load_image(image_path):
    with Image.open(image_path) as arr:
        arr = arr.convert("RGB")
        arr = lp.transform(arr)
        arr = torch.unsqueeze(arr, 0)
        arr = arr.cuda()
        return arr


def load_label(label_path):
    with open (label_path, "r") as fi:
        label = json.load(fi)
    bbox = label["bbox"] if label["class"] == 1 else [[0.]*4]
    bbox = bbox if bbox else [0.0]*4
    return (
        torch.tensor([[label["class"]]], dtype=torch.uint8),
        torch.tensor(bbox, dtype=torch.float32),
    )


def dataset(partition):
    """Yield (image, bounding box) from a partition of the dataset"""

    files = list(glob(f"augmented-data/{partition}/images/*.jpg"))

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


def loca_loss(y_hat, y):
    diff = torch.square(y_hat[:, :2] - y[:, :2])
    summa = torch.sum(diff, dim=-1)

    w_true = y[:, 2] - y[:, 0] 
    h_true = y[:, 3] - y[:, 1] 

    w_pred = y_hat[:, 2] - y_hat[:, 0] 
    h_pred = y_hat[:, 3] - y_hat[:, 1] 

    diff_wh = torch.sum(torch.square(w_true - w_pred) + torch.square(h_true - h_pred), dim=-1)

    return summa + diff_wh


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--display", action="store_true")
    args = parser.parse_args()

    model = IsMattModule(freeze_vgg=False)

    optim = torch.optim.Adam(
        model.parameters(),
        lr=0.000001,
    )

    last_epoch = 0 

    if args.checkpoint is None:
        pass

    elif args.checkpoint and os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])
        optim.load_state_dict(checkpoint["optim_state_dict"])
        last_epoch = checkpoint["epoch"]
        loss = checkpoint["loss"]

        print("+===========================+")
        print("| previous epoch .....", last_epoch)
        print("| batch loss .........", loss)
        print("+===========================+")
        print()
        print()
        print()

        last_epoch += 1

    elif not os.path.exists(args.checkpoint):
        logging.warning(f"checkpoint file {args.checkpoint} does not exist")

    # Samples

    sample = dataset("train")
    sample = [next(sample) for _ in range(5)]


    for epoch in range(last_epoch, last_epoch+args.epochs):


        now = datetime.now()

        print("+===========================+")
        print("| epoch ..............", epoch)


        i = 0
        last_loss = 0.0
        last_loca_loss = 0.0
        last_face_loss = 0.0

        for img, bbox in dataset("train"):

            optim.zero_grad()

            y_hat_face, y_hat_loca = model(img)

            loss = loca_loss(y_hat_loca, bbox[1])

            pos_loss = loca_loss(y_hat_loca, bbox[1])
            fac_loss = torch.nn.functional.binary_cross_entropy(y_hat_face, bbox[0].float())
            loss = pos_loss + 0.5*fac_loss

            loss.backward()
            optim.step()

            # last_loca_loss += loca_loss
            # last_face_loss += face_loss
            last_loss += loss
            i += 1

        # print("| batch loca loss ....", last_loca_loss)
        # print("| batch face loss ....", last_face_loss)
        print("| batch loss .........", last_loss)
        print("| time ............... ", (datetime.now() - now).seconds)
        print("+===========================+")
        print()


        samp = dataset("validate")
        samp = [next(samp) for _ in range(5)]

        # Display some results

        with torch.no_grad():
            for img, bb in samp:
                y_hat_face, y_hat_loca = model(img)
                print("face y hat =>", y_hat_face.cpu().numpy(), bb[0])
                print("loca y hat =>", y_hat_loca.cpu().numpy(), bb[1])

        # Save checkpoint

        model_fname = f"model-{epoch:02d}.pt"

        print(f"Checkpoint @ {model_fname}")

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optim_state_dict": optim.state_dict(),
            "loss": last_loss,
        }, model_fname)


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
