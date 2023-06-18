import torch
from torch import nn


# xs = torch.zeros([64, 3, 224, 224])


class ConvertResnet(nn.Module):

    def __init__(self, in_shape=3):
        super(ConvertResnet, self).__init__()
        self.seq1 = nn.Sequential(
            nn.Conv2d(in_channels=in_shape, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=3, dilation=(1, 1)),
            nn.BatchNorm2d(num_features=64, eps=0.01190501889324183),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1]),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=[1, 1], dilation=(1, 1)),
            nn.BatchNorm2d(num_features=64, eps=0.01190501889324183),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=[1, 1], dilation=(1, 1)),
            nn.BatchNorm2d(num_features=64, eps=0.01190501889324183),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=[1, 1], dilation=(1, 1)),
            nn.BatchNorm2d(num_features=64, eps=0.01190501889324183),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=[1, 1], dilation=(1, 1)),
            nn.BatchNorm2d(num_features=64, eps=0.01190501889324183),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(2, 2), padding=[1, 1], dilation=(1, 1)),
            nn.BatchNorm2d(num_features=128, eps=0.01190501889324183),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=[1, 1], dilation=(1, 1)),
            nn.BatchNorm2d(num_features=128, eps=0.01190501889324183),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=[1, 1], dilation=(1, 1)),
            nn.BatchNorm2d(num_features=128, eps=0.01190501889324183),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=[1, 1], dilation=(1, 1)),
            nn.BatchNorm2d(num_features=128, eps=0.01190501889324183),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=[1, 1], dilation=(1, 1)),
            nn.BatchNorm2d(num_features=128, eps=0.01190501889324183),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=[0, 0], dilation=(1, 1)),
            nn.BatchNorm2d(num_features=256, eps=0.01190501889324183),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=[0, 0], dilation=(1, 1)),
            nn.BatchNorm2d(num_features=256, eps=0.01190501889324183),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=[0, 0], dilation=(1, 1)),
            nn.BatchNorm2d(num_features=256, eps=0.01190501889324183),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=[0, 0], dilation=(1, 1)),
            nn.BatchNorm2d(num_features=256, eps=0.01190501889324183),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=[0, 0], dilation=(1, 1)),
            nn.BatchNorm2d(num_features=256, eps=0.01190501889324183),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=[0, 0], dilation=(1, 1)),
            nn.BatchNorm2d(num_features=512, eps=0.01190501889324183),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=[0, 0], dilation=(1, 1)),
            nn.BatchNorm2d(num_features=512, eps=0.01190501889324183),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=[0, 0], dilation=(1, 1)),
            nn.BatchNorm2d(num_features=512, eps=0.01190501889324183),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=[0, 0], dilation=(1, 1)),
            nn.BatchNorm2d(num_features=512, eps=0.01190501889324183),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=[0, 0], dilation=(1, 1)),
            nn.BatchNorm2d(num_features=512, eps=0.01190501889324183),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=512, out_features=10),
        )

    def forward(self, x_0):
        x_1 = self.seq1(x_0)
        return x_1
