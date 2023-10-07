import albumentations as alb
import cv2
import numpy as np

from torchvision import transforms
from .model import IsMattModule


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

crop = transforms.Compose([
    transforms.Resize(size=256),
    transforms.CenterCrop(size=224),
])


transform = transforms.Compose([
    transforms.Resize(size=256),
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
"""Transform an image into a normalized 224x224 image"""


def cv2_imshow(results):

    images = []

    for img, y, y_hat in results:
        y_hat_face, y_hat_loca = y_hat
        print("... ŷ = ", y_hat_loca.cpu().numpy())
        print("    y = ", y[1].cpu().numpy())
        print()


        arr = img.cpu().numpy()
        arr = arr.squeeze(0)
        arr = np.moveaxis(arr, 0, 2)

        images.append(arr)
    arr = np.concatenate(images, axis=1)
    cv2.imshow("img", arr)
