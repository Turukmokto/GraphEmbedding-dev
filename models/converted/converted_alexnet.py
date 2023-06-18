import torch
from torch import nn


class ConvertedAlexNet(nn.Module):

    def __init__(self):
        super(ConvertedAlexNet, self).__init__()
        self.seq1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0]),
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0]),
            nn.Conv2d(in_channels=192, out_channels=384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0]),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(in_features=1024, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=1000),
        )

    def forward(self, x_0):
        x_1 = self.seq1(x_0)
        return x_1
