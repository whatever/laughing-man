import albumentations as alb
import cv2
import numpy as np
import torch
import torchvision
import torchvision.models.vgg as vgg

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



class IsMattModule(torch.nn.Module):

    shrink = transforms.Compose([
        transforms.Resize(size=224),
        transforms.CenterCrop(size=224),
    ])

    trans =  transforms.Compose(tensorify_arr)

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
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Flatten(),
            torch.nn.Linear(3*3*512, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 4),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.vgg16.features(x)
        return self.face(x), self.loc(x)

    def infer(self, img):
        """Return (face, bb, x) for a PIL image"""
        assert isinstance(img, PIL.Image.Image)

        augmented_img = self.shrink(img.copy().convert("RGB"))
        raise NotImplementedError("TODO: Return augmented image + inference")
        x = self.trans(x)
        bb = [0., 0., 0., 0.]
        face = 0.

        return face, bb, x

    def predict(self, img):
        with torch.no_grad():
            c = crop(img)
            trans = transform(c)
            img = torch.unsqueeze(trans, 0)
            img = img.cuda()
            face, bb = self.forward(img)
        return face, bb, crop


class LaughingPerson(object):
    """Run a loop that captures frames from a camera and blocks faces"""

    def __init__(self, cap):
        self.cap = cap
        self.living = True

    def read(self, *args, **kwargs):
        """Read a frame from device, apply filtering, and return"""
        ret, frame = self.cap.read(*args, **kwargs)
        return ret, frame

    def alive(self):
        """Return device is working"""
        return self.cap.isOpened() and self.living
