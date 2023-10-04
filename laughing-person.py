#!/usr/bin/env python3


import argparse
import cv2
import numpy as np
import torch
import signal

from datetime import timedelta
from datetime import datetime
from train import IsMattModule, transform
from PIL import Image

face_cascade = cv2.CascadeClassifier('capture-images/haarcascade_frontalface_default.xml')

def cleanup(signum, frame):
    raise SystemExit


def capture(device, model):
    cap = cv2.VideoCapture(device)

    org = (00, 50)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255, 255, 255)
    thickness = 2


    checkpoint = datetime.now()

    i = 0

    while True:
        ret, frame = cap.read()

        delta = datetime.now() - checkpoint

        cv2.putText(
            frame,
            f"{delta.seconds + delta.microseconds/1000000:.02f}",
            org,
            font,
            fontScale,
            color,
            thickness,
            cv2.LINE_AA,
        )


        # CV2 Face Tracking
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for x, y, w, h in faces:
            # To draw a rectangle in a face 

            mw = 0
            mh = 0
            x = max(x - mw, 0)
            y = max(y - 2*mh, 0)
            w = w + 2*mw
            h = h + 2*mh

            cv2.rectangle(
                frame,
                (x, y),
                (x+w, y+h),
                (255,255,0),
                2,
            ) 

        # Is Matt face tracking
        with torch.no_grad():

            w, h, _ = frame.shape

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = transform(img).to("cuda")
            img = torch.unsqueeze(img, 0)
            face, bb = model(img)


            face = float(face[:, 0][0].cpu()) == 1

            if face:
                x0, y0, x1, y1 = bb[0].cpu().numpy()
                x0 *= w
                x0 = int(x0)

                x1 *= w
                x1 = int(x1)

                y0 *= h
                y0 = int(y0)

                y1 *= h
                y1 = int(y1)

                cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--capture", type=int)
    args = parser.parse_args()

    signal.signal(signal.SIGINT, cleanup)

    checkpoint = torch.load(args.checkpoint)

    model = IsMattModule()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model = model.to("cuda")


    if args.capture is None:
        print("I do not know what to do :/")
    else:
        capture(args.capture, model)
