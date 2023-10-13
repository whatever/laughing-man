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

from torch.utils.data import Dataset, DataLoader

from collections import defaultdict

from torchvision import transforms

from datetime import datetime
from PIL import Image

from glob import glob

import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

import xD
import xD.data
import xD.model


from xD import DEVICE


torch.set_default_device(xD.DEVICE)


def loca_loss(y_hat, y):
    diff = torch.square(y_hat[:, :2] - y[:, :2])
    summa = torch.sum(diff)

    w_true = y[:, 2] - y[:, 0] 
    h_true = y[:, 3] - y[:, 1] 

    w_pred = y_hat[:, 2] - y_hat[:, 0] 
    h_pred = y_hat[:, 3] - y_hat[:, 1] 

    diff_wh = torch.sum(torch.square(w_true - w_pred)) + torch.sum(torch.square(h_true - h_pred))

    return summa + diff_wh



def display_benchmark(model, samples):
    with torch.no_grad():

        images = []

        for imgs, y in samples:

            o, img = imgs
            o = xD.crop(o)

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
            xD.cv2_imshow(images)
            cv2.waitKey(333)




def main(epochs, checkpoint, display):

    model = xD.model.IsMattModule(freeze_vgg=False)

    optim = torch.optim.Adam(
        model.parameters(),
        lr=0.00001,
    )

    last_epoch = 0 

    if checkpoint is None:
        pass

    elif checkpoint and os.path.exists(checkpoint):
        checkpoint = torch.load(checkpoint)
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

    elif not os.path.exists(checkpoint):
        logging.warning(f"checkpoint file {checkpoint} does not exist")

    if display:
        samp = dataset("validate")
        samp = [next(samp) for _ in range(5)]
        display_benchmark(model, samp)


    for epoch in range(last_epoch, last_epoch+epochs):


        now = datetime.now()

        print("+===================================+")
        print("| epoch ..............", epoch)


        i = 0
        last_loss = 0.0
        last_loca_loss = 0.0
        last_face_loss = 0.0

        data = xD.data.Dataset(
            "augmented-data/train/images/",
            "augmented-data/train/labels/",
        )

        data = DataLoader(data, batch_size=2)

        # XXX: Use command line arg instead here
        for sample in data:

            x = sample["image"]
            face = sample["face"]
            bbox = sample["bbox"]

            optim.zero_grad()

            y_hat_face, y_hat_loca = model(x)

            # pos_loss = torchvision.ops.distance_box_iou_loss(y_hat_loca, bbox)
            pos_loss = loca_loss(y_hat_loca, bbox)
            fac_loss = torch.nn.functional.binary_cross_entropy(y_hat_face, face.float())

            # loss = fac_loss
            loss = pos_loss

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

        if display:
            samp = dataset("validate")
            samp = [next(samp) for _ in range(5)]
            display_benchmark(model, samp)

    logging.info("Saving model %s", checkpoint)

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optim_state_dict": optim.state_dict(),
        "loss": last_loss,
    }, checkpoint)
