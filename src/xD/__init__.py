import albumentations as alb
import cv2
import numpy as np
import torch

from torchvision import transforms
import PIL


def get_label_fname(image_fname):
    return (
        image_fname
        .replace("images", "labels")
        .replace(".jpg", ".json")
    )

CROP_WIDTH = CROP_HEIGHT = 1000

ts = [
    alb.SmallestMaxSize(max_size=min(CROP_WIDTH, CROP_HEIGHT)),
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

large_crop = transforms.Compose([
    transforms.Resize(size=256*3),
    transforms.CenterCrop(size=224*3),
])

crop_arr = [
    transforms.Resize(size=256),
    transforms.CenterCrop(size=224),
]

tensorify_arr = [
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
]


crop = transforms.Compose(crop_arr)
"""Resize smallest size to 256, then crop out the center to 224x224"""


transform = transforms.Compose(tensorify_arr)
"""Transform an image into a normalized 224x224 image"""


def cv2_imshow(results):

    images = []

    for img, y, y_hat in results:
        y_hat_face, y_hat_loca = y_hat

        arr = img.cpu().numpy()
        arr = arr.squeeze(0)
        # arr = np.moveaxis(arr, 0, 2)
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)

        w = h = 224

        bb = y_hat_loca[0].cpu().numpy()
        bb2 = y[1][0].cpu().numpy()
        wh = np.array([w, h, w, h])
        p = [int(x) for x in bb*wh]
        q = [int(x) for x in bb2*wh]

        cv2.rectangle(arr, p[0:2], p[2:4], (255, 0, 255), 3)
        cv2.rectangle(arr, q[0:2], q[2:4], (0, 255, 0), 3)

        images.append(arr)

    arr = cv2.vconcat(images)

    cv2.imshow("img", arr)





