import torch
import torch.nn as nn
import torchvision


class ResNet(nn.Module):
    def __init__(self, num_classes):
        super(ResNet, self).__init__()
        self.net = torchvision.models.resnet18(pretrained=True)
        for param in self.net.parameters():
            param.requires_grad = False
        num_features = self.net.fc.in_features
        self.base = nn.Sequential(*list(self.net.children())[:-1])
        self.last_layer = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.base(x)
        x = x.view(x.size()[0], -1)
        x = self.last_layer(x)
        return x
