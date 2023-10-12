import torch
import torchvision
import torchvision.models.vgg as vgg


from xD import DEVICE


class IsMattModule(torch.nn.Module):

    def __init__(self, freeze_vgg=True, vgg16_weights=vgg.VGG16_Weights.DEFAULT):

        super(IsMattModule, self).__init__()

        vgg16 = torchvision.models.vgg16(weights=vgg16_weights).to(DEVICE)

        for p in vgg16.parameters():
            p.requires_grad = freeze_vgg

        self.features = vgg16.features

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
            torch.nn.Linear(7*7*512, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 4),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.features(x)
        return self.face(x), self.loc(x)


def IsMattQuantizedModel():
    return torch.quantization.quantize_dynamic(
        IsMattModule(),
        {torch.nn.Linear, torch.nn.Conv2d, torch.nn.ReLU, torch.nn.Sigmoid},
        dtype=torch.qint8,
    )
