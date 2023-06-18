from collections import OrderedDict
from typing import Any

import torch
from torch import nn
from torchvision.models.densenet import _densenet

class Net11(nn.Module):
    def __init__(self):
        super(Net11, self).__init__()
        self.conv = nn.Conv2d(3, 64, (2, 2))
        self.log_softmax = nn.LogSoftmax()

    def forward(self, x):
        x = self.conv(x)
        x = self.log_softmax(x)
        return x


class GeneratedModel1(nn.Module):
    def __init__(self):
        super(GeneratedModel1, self).__init__()
        self.seq0 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
                      dilation=(1, 1), groups=1),
        )
        self.seq1 = nn.Sequential(
            nn.BatchNorm2d(num_features=64, eps=1e-05),
            nn.ReLU(),
        )
        self.layers = nn.ModuleList()
        for i in range(1, 62):
            layer = nn.Sequential(
                nn.BatchNorm2d(num_features=64, eps=1e-05),
                nn.ReLU(),
            )
            self.layers.add_module('layer%d' % (i + 1), layer)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(802816, 100)
        )

    def forward(self, x):
        x_0 = self.seq0(x)
        x_1 = self.seq1(x_0)
        for layer in self.layers:
            x_1 = layer(x_0) + x_1
        x_1 = self.fc(x_1)
        return x_1


def GeneratedDensenet(pretrained: bool = False, progress: bool = True, **kwargs: Any):
    return _densenet('GeneratedDensenet', 32, (6, 32, 61, 48), 64, pretrained, progress,
                     **kwargs)


class NaturalSceneClassification(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(

            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            nn.Linear(82944, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 6)
        )

    def forward(self, xb):
        return self.network(xb)

    from torch import nn


class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        input_size = 784
        hidden_sizes = [128, 64]
        output_size = 10

        self.model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                                   nn.ReLU(),
                                   nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                                   nn.ReLU(),
                                   nn.Linear(hidden_sizes[1], output_size),
                                   nn.ReLU())

    def forward(self, x):
        return self.model(x)