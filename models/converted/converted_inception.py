import torch
from torch import nn


class ConvertedInception(nn.Module):

    def __init__(self):
        super(ConvertedInception, self).__init__()
        self.seq1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=32, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=32, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=64, eps=0.001),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding=[0, 0]),
            nn.Conv2d(in_channels=64, out_channels=80, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=80, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=80, out_channels=192, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=192, eps=0.001),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding=[0, 0]),
        )
        self.seq2 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=64, eps=0.001),
            nn.ReLU(),
        )
        self.seq4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=64, eps=0.001),
            nn.ReLU(),
        )
        self.seq6 = nn.Sequential(
            nn.Conv2d(in_channels=288, out_channels=64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=64, eps=0.001),
            nn.ReLU(),
        )
        self.seq8 = nn.Sequential(
            nn.Conv2d(in_channels=288, out_channels=384, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=384, eps=0.001),
            nn.ReLU(),
        )
        self.seq10 = nn.Sequential(
            nn.Conv2d(in_channels=768, out_channels=192, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=192, eps=0.001),
            nn.ReLU(),
        )
        self.seq12 = nn.Sequential(
            nn.Conv2d(in_channels=768, out_channels=192, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=192, eps=0.001),
            nn.ReLU(),
        )
        self.seq14 = nn.Sequential(
            nn.Conv2d(in_channels=768, out_channels=192, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=192, eps=0.001),
            nn.ReLU(),
        )
        self.seq16 = nn.Sequential(
            nn.Conv2d(in_channels=768, out_channels=192, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=192, eps=0.001),
            nn.ReLU(),
        )
        self.seq18 = nn.Sequential(
            nn.Conv2d(in_channels=768, out_channels=192, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=192, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=192, out_channels=320, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=320, eps=0.001),
            nn.ReLU(),
        )
        self.seq20 = nn.Sequential(
            nn.Conv2d(in_channels=1280, out_channels=320, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=320, eps=0.001),
            nn.ReLU(),
        )
        self.seq22 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=320, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=320, eps=0.001),
            nn.ReLU(),
        )
        self.seq23 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=2048, out_features=1000),
        )
        self.seq24 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=384, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=384, eps=0.001),
            nn.ReLU(),
        )
        self.seq25 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=384, eps=0.001),
            nn.ReLU(),
        )
        self.seq27 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=384, eps=0.001),
            nn.ReLU(),
        )
        self.seq28 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=448, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=448, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=448, out_channels=384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=384, eps=0.001),
            nn.ReLU(),
        )
        self.seq29 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=384, eps=0.001),
            nn.ReLU(),
        )
        self.seq31 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=384, eps=0.001),
            nn.ReLU(),
        )
        self.seq32 = nn.Sequential(
            nn.AvgPool2d(kernel_size=[3, 3], stride=[1, 1], padding=[0, 0]),
            nn.Conv2d(in_channels=2048, out_channels=192, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=192, eps=0.001),
            nn.ReLU(),
        )
        self.seq33 = nn.Sequential(
            nn.Conv2d(in_channels=1280, out_channels=384, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=384, eps=0.001),
            nn.ReLU(),
        )
        self.seq34 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=384, eps=0.001),
            nn.ReLU(),
        )
        self.seq36 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=384, eps=0.001),
            nn.ReLU(),
        )
        self.seq37 = nn.Sequential(
            nn.Conv2d(in_channels=1280, out_channels=448, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=448, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=448, out_channels=384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=384, eps=0.001),
            nn.ReLU(),
        )
        self.seq38 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=384, eps=0.001),
            nn.ReLU(),
        )
        self.seq40 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=384, eps=0.001),
            nn.ReLU(),
        )
        self.seq41 = nn.Sequential(
            nn.AvgPool2d(kernel_size=[3, 3], stride=[1, 1], padding=[0, 0]),
            nn.Conv2d(in_channels=1280, out_channels=192, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=192, eps=0.001),
            nn.ReLU(),
        )
        self.seq42 = nn.Sequential(
            nn.Conv2d(in_channels=768, out_channels=192, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=192, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=192, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=192, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=192, eps=0.001),
            nn.ReLU(),
        )
        self.seq43 = nn.Sequential(
            nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding=[0, 0]),
        )
        self.seq44 = nn.Sequential(
            nn.Conv2d(in_channels=768, out_channels=192, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=192, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=192, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=192, eps=0.001),
            nn.ReLU(),
        )
        self.seq45 = nn.Sequential(
            nn.Conv2d(in_channels=768, out_channels=192, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=192, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=192, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=192, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=192, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=192, eps=0.001),
            nn.ReLU(),
        )
        self.seq46 = nn.Sequential(
            nn.AvgPool2d(kernel_size=[3, 3], stride=[1, 1], padding=[0, 0]),
            nn.Conv2d(in_channels=768, out_channels=192, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=192, eps=0.001),
            nn.ReLU(),
        )
        self.seq47 = nn.Sequential(
            nn.Conv2d(in_channels=768, out_channels=160, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=160, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=160, out_channels=160, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=160, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=160, out_channels=192, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=192, eps=0.001),
            nn.ReLU(),
        )
        self.seq48 = nn.Sequential(
            nn.Conv2d(in_channels=768, out_channels=160, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=160, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=160, out_channels=160, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=160, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=160, out_channels=160, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=160, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=160, out_channels=160, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=160, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=160, out_channels=192, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=192, eps=0.001),
            nn.ReLU(),
        )
        self.seq49 = nn.Sequential(
            nn.AvgPool2d(kernel_size=[3, 3], stride=[1, 1], padding=[0, 0]),
            nn.Conv2d(in_channels=768, out_channels=192, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=192, eps=0.001),
            nn.ReLU(),
        )
        self.seq50 = nn.Sequential(
            nn.Conv2d(in_channels=768, out_channels=160, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=160, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=160, out_channels=160, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=160, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=160, out_channels=192, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=192, eps=0.001),
            nn.ReLU(),
        )
        self.seq51 = nn.Sequential(
            nn.Conv2d(in_channels=768, out_channels=160, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=160, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=160, out_channels=160, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=160, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=160, out_channels=160, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=160, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=160, out_channels=160, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=160, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=160, out_channels=192, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=192, eps=0.001),
            nn.ReLU(),
        )
        self.seq52 = nn.Sequential(
            nn.AvgPool2d(kernel_size=[3, 3], stride=[1, 1], padding=[0, 0]),
            nn.Conv2d(in_channels=768, out_channels=192, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=192, eps=0.001),
            nn.ReLU(),
        )
        self.seq53 = nn.Sequential(
            nn.Conv2d(in_channels=768, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=192, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=192, eps=0.001),
            nn.ReLU(),
        )
        self.seq54 = nn.Sequential(
            nn.Conv2d(in_channels=768, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=192, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=192, eps=0.001),
            nn.ReLU(),
        )
        self.seq55 = nn.Sequential(
            nn.AvgPool2d(kernel_size=[3, 3], stride=[1, 1], padding=[0, 0]),
            nn.Conv2d(in_channels=768, out_channels=192, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=192, eps=0.001),
            nn.ReLU(),
        )
        self.seq56 = nn.Sequential(
            nn.Conv2d(in_channels=288, out_channels=64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=64, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=96, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=96, eps=0.001),
            nn.ReLU(),
        )
        self.seq57 = nn.Sequential(
            nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding=[0, 0]),
        )
        self.seq58 = nn.Sequential(
            nn.Conv2d(in_channels=288, out_channels=48, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=48, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=64, eps=0.001),
            nn.ReLU(),
        )
        self.seq59 = nn.Sequential(
            nn.Conv2d(in_channels=288, out_channels=64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=64, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=96, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=96, eps=0.001),
            nn.ReLU(),
        )
        self.seq60 = nn.Sequential(
            nn.AvgPool2d(kernel_size=[3, 3], stride=[1, 1], padding=[0, 0]),
            nn.Conv2d(in_channels=288, out_channels=64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=64, eps=0.001),
            nn.ReLU(),
        )
        self.seq61 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=48, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=48, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=64, eps=0.001),
            nn.ReLU(),
        )
        self.seq62 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=64, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=96, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=96, eps=0.001),
            nn.ReLU(),
        )
        self.seq63 = nn.Sequential(
            nn.AvgPool2d(kernel_size=[3, 3], stride=[1, 1], padding=[0, 0]),
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=64, eps=0.001),
            nn.ReLU(),
        )
        self.seq64 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=48, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=48, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=64, eps=0.001),
            nn.ReLU(),
        )
        self.seq65 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=64, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=96, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=96, eps=0.001),
            nn.ReLU(),
        )
        self.seq66 = nn.Sequential(
            nn.AvgPool2d(kernel_size=[3, 3], stride=[1, 1], padding=[0, 0]),
            nn.Conv2d(in_channels=192, out_channels=32, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=32, eps=0.001),
            nn.ReLU(),
        )

    def forward(self, x_0):
        x_1 = self.seq1(x_0)
        x_66 = torch.nn.functional.pad(x_1, [1, 1, 1, 1, 0, 0, 0, 0])
        x_66 = self.seq66(x_66)
        x_65 = self.seq65(x_1)
        x_64 = self.seq64(x_1)
        x_2 = self.seq2(x_1)
        x_3 = torch.cat([x_2, x_64, x_65, x_66], 1)
        x_63 = torch.nn.functional.pad(x_3, [1, 1, 1, 1, 0, 0, 0, 0])
        x_63 = self.seq63(x_63)
        x_62 = self.seq62(x_3)
        x_61 = self.seq61(x_3)
        x_4 = self.seq4(x_3)
        x_5 = torch.cat([x_4, x_61, x_62, x_63], 1)
        x_60 = torch.nn.functional.pad(x_5, [1, 1, 1, 1, 0, 0, 0, 0])
        x_60 = self.seq60(x_60)
        x_59 = self.seq59(x_5)
        x_58 = self.seq58(x_5)
        x_6 = self.seq6(x_5)
        x_7 = torch.cat([x_6, x_58, x_59, x_60], 1)
        x_57 = self.seq57(x_7)
        x_56 = self.seq56(x_7)
        x_8 = self.seq8(x_7)
        x_9 = torch.cat([x_8, x_56, x_57], 1)
        x_55 = torch.nn.functional.pad(x_9, [1, 1, 1, 1, 0, 0, 0, 0])
        x_55 = self.seq55(x_55)
        x_54 = self.seq54(x_9)
        x_53 = self.seq53(x_9)
        x_10 = self.seq10(x_9)
        x_11 = torch.cat([x_10, x_53, x_54, x_55], 1)
        x_52 = torch.nn.functional.pad(x_11, [1, 1, 1, 1, 0, 0, 0, 0])
        x_52 = self.seq52(x_52)
        x_51 = self.seq51(x_11)
        x_50 = self.seq50(x_11)
        x_12 = self.seq12(x_11)
        x_13 = torch.cat([x_12, x_50, x_51, x_52], 1)
        x_49 = torch.nn.functional.pad(x_13, [1, 1, 1, 1, 0, 0, 0, 0])
        x_49 = self.seq49(x_49)
        x_48 = self.seq48(x_13)
        x_47 = self.seq47(x_13)
        x_14 = self.seq14(x_13)
        x_15 = torch.cat([x_14, x_47, x_48, x_49], 1)
        x_46 = torch.nn.functional.pad(x_15, [1, 1, 1, 1, 0, 0, 0, 0])
        x_46 = self.seq46(x_46)
        x_45 = self.seq45(x_15)
        x_44 = self.seq44(x_15)
        x_16 = self.seq16(x_15)
        x_17 = torch.cat([x_16, x_44, x_45, x_46], 1)
        x_43 = self.seq43(x_17)
        x_42 = self.seq42(x_17)
        x_18 = self.seq18(x_17)
        x_19 = torch.cat([x_18, x_42, x_43], 1)
        x_41 = torch.nn.functional.pad(x_19, [1, 1, 1, 1, 0, 0, 0, 0])
        x_41 = self.seq41(x_41)
        x_37 = self.seq37(x_19)
        x_40 = self.seq40(x_37)
        x_38 = self.seq38(x_37)
        x_39 = torch.cat([x_38, x_40], 1)
        x_33 = self.seq33(x_19)
        x_36 = self.seq36(x_33)
        x_34 = self.seq34(x_33)
        x_35 = torch.cat([x_34, x_36], 1)
        x_20 = self.seq20(x_19)
        x_21 = torch.cat([x_20, x_35, x_39, x_41], 1)
        x_32 = torch.nn.functional.pad(x_21, [1, 1, 1, 1, 0, 0, 0, 0])
        x_32 = self.seq32(x_32)
        x_28 = self.seq28(x_21)
        x_31 = self.seq31(x_28)
        x_29 = self.seq29(x_28)
        x_30 = torch.cat([x_29, x_31], 1)
        x_24 = self.seq24(x_21)
        x_27 = self.seq27(x_24)
        x_25 = self.seq25(x_24)
        x_26 = torch.cat([x_25, x_27], 1)
        x_22 = self.seq22(x_21)
        x_23 = torch.cat([x_22, x_26, x_30, x_32], 1)
        x_23 = self.seq23(x_23)
        return x_23
