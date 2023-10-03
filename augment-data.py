#!/usr/bin/env python3


import albumentations as alb
import cv2
import matplotlib.pyplot as plot
import numpy as np
import json
import numpy
import os.path
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


def show_augmented_image(aug):
    img = aug["image"].copy()
    for bb in aug["bboxes"]:
        cv2.rectangle(
            img,
            tuple(np.multiply(aug["bboxes"][0][0:2], CROP_WIDTH).astype(int)),
            tuple(np.multiply(aug["bboxes"][0][2:], CROP_HEIGHT).astype(int)),
            (0, 255, 0),
            2,
        )
    show_image(img)


def get_standardized_bbox(label_fname):
    """Convert a bounding box to the standard format"""

    with open(label_fname, "r") as f:
        label = json.load(f)

    shapes = label["shapes"]

    assert len(shapes) == 1

    shape = shapes[0]

    return [
        bbox[0],
        bbox[1],
        bbox[0] + bbox[2],
        bbox[1] + bbox[3],
    ]


def get_label_fname(image_fname):
    return (
        image_fname
        .replace("images", "labels")
        .replace(".jpg", ".json")
    )


def get_image_and_label(image_fname):

    label_fname = get_label_fname(image_fname)

    with open(label_fname, "r") as fi:
        label = json.load(fi)

    shape = label.get("shapes", [None])[0]


    img = Image.open(image_fname).convert("RGB")
    w, h = img.size


    if not shape:
        return {
            "image_fname": image_fname,
            "label_fname": label_fname,
            "bbox": [0, 0, 0.000001,  0.000001],
            "class": 0,
        }, img

    p0, p1 = shape["points"]

    coords = [
        min(p0[0], p1[0]),
        min(p0[1], p1[1]),
        max(p0[0], p1[0]),
        max(p0[1], p1[1]),
    ]

    bb = [
        coords[0] / w,
        coords[1] / h,
        coords[2] / w,
        coords[3] / h,
    ]

    class_label = int(shape["label"] == "matt-face")

    res = {
        "image_fname": image_fname,
        "label_fname": label_fname,
        "bbox": bb,
        "class": class_label,
    }

    return res, img



if __name__ == "__main__":


    """
    for fname in glob("data/test/labels/*.json"):
        with open(fname, "r") as fi:
            label = json.load(fi)

        shapes = label["shapes"]
        assert len(shapes) == 1


        shape = shapes[0]

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

        show_augmented_image(augmented_image)


    cv2.destroyAllWindows()
    """


    # Modify data


    for partition in ["train", "test", "validate"]:
        for image_fname in glob(f"data/{partition}/images/*.jpg"):
            res, img = get_image_and_label(image_fname)

            for i in range(60):
                aug = augmentor(
                    image=np.array(img),
                    bboxes=[res["bbox"]],
                    class_labels=["face"],
                )

                aug_basename = "{}-{}".format(
                    os.path.basename(image_fname).replace(".jpg", ""),
                    i,
                )

                aug_image_fname = os.path.join(
                    "aug_data",
                    partition,
                    "images",
                    f"{aug_basename}.jpg",
                )

                aug_label_fname = os.path.join(
                    "aug_data",
                    partition,
                    "labels",
                    f"{aug_basename}.json",
                )

                annotation = {
                    "bbox": res["bbox"],
                    "class": res["class"],
                }

                aug_img = Image.fromarray(aug["image"])
                aug_img.save(aug_image_fname)

                with open(aug_label_fname, "w") as f:
                    json.dump(annotation, f)


    raise SystemExit

    for partition in ["train", "test", "validate"]:
        for label_fname in glob(f"data/{partition}/labels/*.json"):
            image_fname = label_fname.replace("labels", "images").replace(".json", ".jpg")
            aug_fname = os.path.join("aug_data", partition, "labels", os.path.basename(label_fname))



            if not os.path.exists(image_fname):
                annotation["bbox"] = [0]*4
                annotation["class"] = 0

            with open(aug_fname, "w") as f:
                # json.dump(aug_
                pass

            print(label_fname)
            print(image_fname)
            print(aug_fname)




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
