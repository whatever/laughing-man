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

from collections import defaultdict

from torchvision import transforms

from datetime import datetime
from PIL import Image

torch.set_default_device('cuda')

from glob import glob

import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')


def get_label_fname(image_fname):
    return (
        image_fname
        .replace("images", "labels")
        .replace(".jpg", ".json")
    )


def load_image(image_path):
    with Image.open(image_path) as img:
        arr = img.copy()
        arr = arr.convert("RGB")
        arr = lp.crop(arr)
        arr = lp.transform(arr)
        arr = torch.unsqueeze(arr, 0)
        arr = arr.cuda()
        return img, arr


def load_label(label_path):
    if not os.path.exists(label_path):
        return (
            torch.tensor([[0]], dtype=torch.uint8),
            torch.tensor([[0.]*4], dtype=torch.float32),
        )
    with open (label_path, "r") as fi:
        label = json.load(fi)
    bbox = label["bbox"] if label["class"] == 1 else [[0.]*4]
    bbox = bbox if bbox else [0.0]*4
    return (
        torch.tensor([[label["class"]]], dtype=torch.uint8),
        torch.tensor(bbox, dtype=torch.float32),
    )


def dataset(partition, n=None):
    """Yield (image, bounding box) from a partition of the dataset"""

    files = list(glob(f"augmented-data/{partition}/images/*.jpg"))

    if n is not None:
        files = files[:n]

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



def loca_loss(y_hat, y):
    diff = torch.square(y_hat[:, :2] - y[:, :2])
    summa = torch.sum(diff, dim=-1)

    w_true = y[:, 2] - y[:, 0] 
    h_true = y[:, 3] - y[:, 1] 

    w_pred = y_hat[:, 2] - y_hat[:, 0] 
    h_pred = y_hat[:, 3] - y_hat[:, 1] 

    diff_wh = torch.sum(torch.square(w_true - w_pred), dim=-1) + torch.sum(torch.square(h_true - h_pred), dim=-1)

    return summa + diff_wh



def display_benchmark(model, samples):
    with torch.no_grad():

        images = []

        for imgs, y in samples:

            o, img = imgs
            o = lp.crop(o)

            y_hat = model(img)

            # print(f"y ........ {y}")
            # print(f"y_hat .... {y_hat}")
            # print()

            images.append((
                torch.tensor(np.array(o)[None, :, :, :]),
                [y[0], y[1]],
                [y_hat[0], y_hat[1]],
            ))

        for i in range(2):
            lp.cv2_imshow(images)
            cv2.waitKey(333)




if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--display", action="store_true")
    args = parser.parse_args()

    model = lp.IsMattModule(freeze_vgg=False)

    optim = torch.optim.Adam(
        model.parameters(),
        lr=0.00001,
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

    if args.display:
        samp = dataset("validate")
        samp = [next(samp) for _ in range(5)]
        display_benchmark(model, samp)


    for epoch in range(last_epoch, last_epoch+args.epochs):


        now = datetime.now()

        print("+===================================+")
        print("| epoch ..............", epoch)


        i = 0
        last_loss = 0.0
        last_loca_loss = 0.0
        last_face_loss = 0.0

        for imgs, bbox in dataset("train"):

            _, img = imgs

            optim.zero_grad()

            y_hat_face, y_hat_loca = model(img)

            # pos_loss = loca_loss(y_hat_loca, bbox[1])
            pos_loss = torchvision.ops.distance_box_iou_loss(y_hat_loca, bbox[1])
            fac_loss = torch.nn.functional.binary_cross_entropy(y_hat_face, bbox[0].float())
            loss = 0.75*pos_loss + 0.25*fac_loss

            loss.backward()
            optim.step()

            last_loss += loss
            i += 1

        # print("| batch face loss ....", last_face_loss)
        print(f"| batch loss ......... {float(last_loss):.2f}")
        print(f"| time ............... {(datetime.now() - now).seconds}")
        print("+===================================+")
        print()


        # Display some results

        if args.display:
            samp = dataset("validate")
            samp = [next(samp) for _ in range(5)]
            display_benchmark(model, samp)

    logging.info("Saving model %s", args.checkpoint)

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optim_state_dict": optim.state_dict(),
        "loss": last_loss,
    }, args.checkpoint)
