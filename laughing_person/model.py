import torch
import torchvision
import torchvision.models.vgg as vgg

class IsMattModule(torch.nn.Module):

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

    def predict(self, img):
        with torch.no_grad():
            crop = lp.crop(img)
            crop = lp.crop(crop)
            crop = lp.crop(crop)
            crop = lp.crop(crop)
            crop = lp.crop(crop)

            trans = lp.transform(crop)
            img = torch.unsqueeze(trans, 0)
            img = img.cuda()
            face, bb = self.forward(img)
        return face, bb, crop
