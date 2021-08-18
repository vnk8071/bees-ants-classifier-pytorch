import torch
import torch.nn as nn
import torchvision


class Backbone(nn.Module):
    def __init__(self, num_classes, is_pretrained=True):
        super(Backbone, self).__init__()
        self.net = torchvision.models.resnet18(pretrained=is_pretrained)
        for param in self.net.parameters():
            param.requires_grad = False
        num_features = self.net.fc.in_features
        self.net.fc = nn.Sequential(
            nn.Linear(num_features, num_classes), nn.Sigmoid())

    def forward(self, x):
        x = self.net(x)
        return x


class ResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = Backbone(num_classes, True)
        # TODO: develop your retinal model

    def forward(self, images):
        return self.backbone(images)
