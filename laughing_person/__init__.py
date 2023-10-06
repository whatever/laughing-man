import albumentations as alb
from torchvision import transforms


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
