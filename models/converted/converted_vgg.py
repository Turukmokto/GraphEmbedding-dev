import torch
from torch import nn


class ConvertedVGG(nn.Module):

    def __init__(self):
        super(ConvertedVGG, self).__init__()
        self.seq1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=64, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=64, eps=1e-05),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0]),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0]),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0]),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0]),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0]),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(in_features=25088, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=1000),
        )

    def forward(self, x_0):
        x_1 = self.seq1(x_0)
        return x_1
