import cv2
import numpy as np
import PIL
import torch

from xD.model import IsMattModule
import xD


DEVICE = "cpu"


class LaughingPerson(object):
    """Run a loop that captures frames from a camera and blocks faces"""

    def __init__(self, cap, model_path):
        """Initialize"""
        self.model = IsMattModule().to(DEVICE)
        state = torch.load(
            model_path,
            map_location=DEVICE,
        )
        self.model.load_state_dict(state["model_state_dict"])
        self.model.eval()
        self.cap = cap
        self.living = True
        self.last_raw_frame = None

    def read(self, *args, **kwargs):
        """Read a frame from device, apply filtering, and return"""
        ret, frame = self.cap.read(*args, **kwargs)

        self.last_read = (ret, frame)

        if not ret:
            return ret, frame

        # CV2 BGR -> PIL RGB
        x_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        x_img = PIL.Image.fromarray(x_img)

        x_large_img = xD.large_crop(x_img)
        x_img = xD.crop(x_img)

        # PIL RGB -> Tensor
        with torch.no_grad():
            x = xD.transform(x_img)
            x = torch.unsqueeze(x, 0).to(DEVICE)
            face, bbox = self.model(x)
            face = float(face[0][0])

            w, h = x_large_img.size
            bbox = bbox.cpu() * np.array([w, h, w, h])
            bbox = [int(v) for v in bbox[0]]


        # Tensor -> CV2 BGR
        y_img = np.array(x_large_img)
        y_img = cv2.cvtColor(y_img, cv2.COLOR_RGB2BGR)

        if face > 0.95:
            cv2.rectangle(
                y_img,
                bbox[:2],
                bbox[2:],
                (0, 255, 255),
                2,
            )

        put_text_args = [
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            1,
        ]

        cv2.flip(y_img, 1)

        cv2.putText(
            y_img,
            f"face ... {face:.2f}",
            (0, 20),
            *put_text_args,
        )

        return ret, y_img

    def alive(self):
        """Return device is working"""
        return self.cap.isOpened() and self.living
