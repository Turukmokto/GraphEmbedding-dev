import torch
from torch import nn


class ConvertedDenseNet(nn.Module):

    def __init__(self):
        super(ConvertedDenseNet, self).__init__()
        self.seq1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=64, eps=1e-05),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding=[1, 1]),
        )
        self.seq2 = nn.Sequential(
            nn.BatchNorm2d(num_features=64, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq3 = nn.Sequential(
            nn.BatchNorm2d(num_features=96, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=96, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq4 = nn.Sequential(
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq5 = nn.Sequential(
            nn.BatchNorm2d(num_features=160, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=160, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq6 = nn.Sequential(
            nn.BatchNorm2d(num_features=192, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=192, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq7 = nn.Sequential(
            nn.BatchNorm2d(num_features=224, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=224, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq8 = nn.Sequential(
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
        )
        self.seq9 = nn.Sequential(
            nn.AvgPool2d(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0]),
        )
        self.seq10 = nn.Sequential(
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq11 = nn.Sequential(
            nn.BatchNorm2d(num_features=160, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=160, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq12 = nn.Sequential(
            nn.BatchNorm2d(num_features=192, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=192, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq13 = nn.Sequential(
            nn.BatchNorm2d(num_features=224, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=224, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq14 = nn.Sequential(
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq15 = nn.Sequential(
            nn.BatchNorm2d(num_features=288, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=288, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq16 = nn.Sequential(
            nn.BatchNorm2d(num_features=320, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=320, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq17 = nn.Sequential(
            nn.BatchNorm2d(num_features=352, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=352, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq18 = nn.Sequential(
            nn.BatchNorm2d(num_features=384, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq19 = nn.Sequential(
            nn.BatchNorm2d(num_features=416, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=416, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq20 = nn.Sequential(
            nn.BatchNorm2d(num_features=448, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=448, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq21 = nn.Sequential(
            nn.BatchNorm2d(num_features=480, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=480, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq22 = nn.Sequential(
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
        )
        self.seq23 = nn.Sequential(
            nn.AvgPool2d(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0]),
        )
        self.seq24 = nn.Sequential(
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq25 = nn.Sequential(
            nn.BatchNorm2d(num_features=288, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=288, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq26 = nn.Sequential(
            nn.BatchNorm2d(num_features=320, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=320, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq27 = nn.Sequential(
            nn.BatchNorm2d(num_features=352, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=352, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq28 = nn.Sequential(
            nn.BatchNorm2d(num_features=384, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq29 = nn.Sequential(
            nn.BatchNorm2d(num_features=416, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=416, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq30 = nn.Sequential(
            nn.BatchNorm2d(num_features=448, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=448, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq31 = nn.Sequential(
            nn.BatchNorm2d(num_features=480, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=480, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq32 = nn.Sequential(
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq33 = nn.Sequential(
            nn.BatchNorm2d(num_features=544, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=544, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq34 = nn.Sequential(
            nn.BatchNorm2d(num_features=576, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=576, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq35 = nn.Sequential(
            nn.BatchNorm2d(num_features=608, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=608, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq36 = nn.Sequential(
            nn.BatchNorm2d(num_features=640, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=640, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq37 = nn.Sequential(
            nn.BatchNorm2d(num_features=672, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=672, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq38 = nn.Sequential(
            nn.BatchNorm2d(num_features=704, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=704, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq39 = nn.Sequential(
            nn.BatchNorm2d(num_features=736, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=736, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq40 = nn.Sequential(
            nn.BatchNorm2d(num_features=768, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=768, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq41 = nn.Sequential(
            nn.BatchNorm2d(num_features=800, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=800, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq42 = nn.Sequential(
            nn.BatchNorm2d(num_features=832, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=832, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq43 = nn.Sequential(
            nn.BatchNorm2d(num_features=864, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=864, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq44 = nn.Sequential(
            nn.BatchNorm2d(num_features=896, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=896, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq45 = nn.Sequential(
            nn.BatchNorm2d(num_features=928, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=928, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq46 = nn.Sequential(
            nn.BatchNorm2d(num_features=960, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=960, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq47 = nn.Sequential(
            nn.BatchNorm2d(num_features=992, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=992, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq48 = nn.Sequential(
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=1024, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq49 = nn.Sequential(
            nn.BatchNorm2d(num_features=1056, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=1056, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq50 = nn.Sequential(
            nn.BatchNorm2d(num_features=1088, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=1088, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq51 = nn.Sequential(
            nn.BatchNorm2d(num_features=1120, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=1120, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq52 = nn.Sequential(
            nn.BatchNorm2d(num_features=1152, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=1152, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq53 = nn.Sequential(
            nn.BatchNorm2d(num_features=1184, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=1184, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq54 = nn.Sequential(
            nn.BatchNorm2d(num_features=1216, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=1216, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq55 = nn.Sequential(
            nn.BatchNorm2d(num_features=1248, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=1248, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq56 = nn.Sequential(
            nn.BatchNorm2d(num_features=1280, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=1280, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq57 = nn.Sequential(
            nn.BatchNorm2d(num_features=1312, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=1312, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq58 = nn.Sequential(
            nn.BatchNorm2d(num_features=1344, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=1344, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq59 = nn.Sequential(
            nn.BatchNorm2d(num_features=1376, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=1376, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq60 = nn.Sequential(
            nn.BatchNorm2d(num_features=1408, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=1408, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq61 = nn.Sequential(
            nn.BatchNorm2d(num_features=1440, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=1440, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq62 = nn.Sequential(
            nn.BatchNorm2d(num_features=1472, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=1472, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq63 = nn.Sequential(
            nn.BatchNorm2d(num_features=1504, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=1504, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq64 = nn.Sequential(
            nn.BatchNorm2d(num_features=1536, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=1536, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq65 = nn.Sequential(
            nn.BatchNorm2d(num_features=1568, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=1568, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq66 = nn.Sequential(
            nn.BatchNorm2d(num_features=1600, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=1600, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq67 = nn.Sequential(
            nn.BatchNorm2d(num_features=1632, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=1632, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq68 = nn.Sequential(
            nn.BatchNorm2d(num_features=1664, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=1664, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq69 = nn.Sequential(
            nn.BatchNorm2d(num_features=1696, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=1696, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq70 = nn.Sequential(
            nn.BatchNorm2d(num_features=1728, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=1728, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq71 = nn.Sequential(
            nn.BatchNorm2d(num_features=1760, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=1760, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq72 = nn.Sequential(
            nn.BatchNorm2d(num_features=1792, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=1792, out_channels=896, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
        )
        self.seq73 = nn.Sequential(
            nn.AvgPool2d(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0]),
        )
        self.seq74 = nn.Sequential(
            nn.BatchNorm2d(num_features=896, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=896, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq75 = nn.Sequential(
            nn.BatchNorm2d(num_features=928, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=928, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq76 = nn.Sequential(
            nn.BatchNorm2d(num_features=960, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=960, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq77 = nn.Sequential(
            nn.BatchNorm2d(num_features=992, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=992, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq78 = nn.Sequential(
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=1024, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq79 = nn.Sequential(
            nn.BatchNorm2d(num_features=1056, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=1056, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq80 = nn.Sequential(
            nn.BatchNorm2d(num_features=1088, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=1088, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq81 = nn.Sequential(
            nn.BatchNorm2d(num_features=1120, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=1120, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq82 = nn.Sequential(
            nn.BatchNorm2d(num_features=1152, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=1152, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq83 = nn.Sequential(
            nn.BatchNorm2d(num_features=1184, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=1184, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq84 = nn.Sequential(
            nn.BatchNorm2d(num_features=1216, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=1216, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq85 = nn.Sequential(
            nn.BatchNorm2d(num_features=1248, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=1248, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq86 = nn.Sequential(
            nn.BatchNorm2d(num_features=1280, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=1280, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq87 = nn.Sequential(
            nn.BatchNorm2d(num_features=1312, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=1312, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq88 = nn.Sequential(
            nn.BatchNorm2d(num_features=1344, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=1344, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq89 = nn.Sequential(
            nn.BatchNorm2d(num_features=1376, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=1376, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq90 = nn.Sequential(
            nn.BatchNorm2d(num_features=1408, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=1408, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq91 = nn.Sequential(
            nn.BatchNorm2d(num_features=1440, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=1440, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq92 = nn.Sequential(
            nn.BatchNorm2d(num_features=1472, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=1472, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq93 = nn.Sequential(
            nn.BatchNorm2d(num_features=1504, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=1504, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq94 = nn.Sequential(
            nn.BatchNorm2d(num_features=1536, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=1536, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq95 = nn.Sequential(
            nn.BatchNorm2d(num_features=1568, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=1568, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq96 = nn.Sequential(
            nn.BatchNorm2d(num_features=1600, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=1600, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq97 = nn.Sequential(
            nn.BatchNorm2d(num_features=1632, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=1632, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq98 = nn.Sequential(
            nn.BatchNorm2d(num_features=1664, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=1664, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq99 = nn.Sequential(
            nn.BatchNorm2d(num_features=1696, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=1696, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq100 = nn.Sequential(
            nn.BatchNorm2d(num_features=1728, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=1728, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq101 = nn.Sequential(
            nn.BatchNorm2d(num_features=1760, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=1760, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq102 = nn.Sequential(
            nn.BatchNorm2d(num_features=1792, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=1792, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq103 = nn.Sequential(
            nn.BatchNorm2d(num_features=1824, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=1824, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq104 = nn.Sequential(
            nn.BatchNorm2d(num_features=1856, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=1856, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq105 = nn.Sequential(
            nn.BatchNorm2d(num_features=1888, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=1888, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        )
        self.seq106 = nn.Sequential(
            nn.BatchNorm2d(num_features=1920, eps=1e-05),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=1920, out_features=1000),
        )

    def forward(self, x_0):
        x_1 = self.seq1(x_0)
        x_2 = torch.cat([x_1], 1)
        x_2 = self.seq2(x_2)
        x_3 = torch.cat([x_2, x_1], 1)
        x_3 = self.seq3(x_3)
        x_4 = torch.cat([x_3, x_2, x_1], 1)
        x_4 = self.seq4(x_4)
        x_5 = torch.cat([x_4, x_3, x_2, x_1], 1)
        x_5 = self.seq5(x_5)
        x_6 = torch.cat([x_5, x_4, x_3, x_2, x_1], 1)
        x_6 = self.seq6(x_6)
        x_7 = torch.cat([x_6, x_5, x_4, x_3, x_2, x_1], 1)
        x_7 = self.seq7(x_7)
        x_8 = torch.cat([x_7, x_6, x_5, x_4, x_3, x_2, x_1], 1)
        x_8 = self.seq8(x_8)
        x_9 = self.seq9(x_8)
        x_10 = torch.cat([x_9], 1)
        x_10 = self.seq10(x_10)
        x_11 = torch.cat([x_10, x_9], 1)
        x_11 = self.seq11(x_11)
        x_12 = torch.cat([x_11, x_10, x_9], 1)
        x_12 = self.seq12(x_12)
        x_13 = torch.cat([x_12, x_11, x_10, x_9], 1)
        x_13 = self.seq13(x_13)
        x_14 = torch.cat([x_13, x_12, x_11, x_10, x_9], 1)
        x_14 = self.seq14(x_14)
        x_15 = torch.cat([x_14, x_13, x_12, x_11, x_10, x_9], 1)
        x_15 = self.seq15(x_15)
        x_16 = torch.cat([x_15, x_14, x_13, x_12, x_11, x_10, x_9], 1)
        x_16 = self.seq16(x_16)
        x_17 = torch.cat([x_16, x_15, x_14, x_13, x_12, x_11, x_10, x_9], 1)
        x_17 = self.seq17(x_17)
        x_18 = torch.cat([x_17, x_16, x_15, x_14, x_13, x_12, x_11, x_10, x_9], 1)
        x_18 = self.seq18(x_18)
        x_19 = torch.cat([x_18, x_17, x_16, x_15, x_14, x_13, x_12, x_11, x_10, x_9], 1)
        x_19 = self.seq19(x_19)
        x_20 = torch.cat([x_19, x_18, x_17, x_16, x_15, x_14, x_13, x_12, x_11, x_10, x_9], 1)
        x_20 = self.seq20(x_20)
        x_21 = torch.cat([x_20, x_19, x_18, x_17, x_16, x_15, x_14, x_13, x_12, x_11, x_10, x_9], 1)
        x_21 = self.seq21(x_21)
        x_22 = torch.cat([x_21, x_20, x_19, x_18, x_17, x_16, x_15, x_14, x_13, x_12, x_11, x_10, x_9], 1)
        x_22 = self.seq22(x_22)
        x_23 = self.seq23(x_22)
        x_24 = torch.cat([x_23], 1)
        x_24 = self.seq24(x_24)
        x_25 = torch.cat([x_24, x_23], 1)
        x_25 = self.seq25(x_25)
        x_26 = torch.cat([x_25, x_24, x_23], 1)
        x_26 = self.seq26(x_26)
        x_27 = torch.cat([x_26, x_25, x_24, x_23], 1)
        x_27 = self.seq27(x_27)
        x_28 = torch.cat([x_27, x_26, x_25, x_24, x_23], 1)
        x_28 = self.seq28(x_28)
        x_29 = torch.cat([x_28, x_27, x_26, x_25, x_24, x_23], 1)
        x_29 = self.seq29(x_29)
        x_30 = torch.cat([x_29, x_28, x_27, x_26, x_25, x_24, x_23], 1)
        x_30 = self.seq30(x_30)
        x_31 = torch.cat([x_30, x_29, x_28, x_27, x_26, x_25, x_24, x_23], 1)
        x_31 = self.seq31(x_31)
        x_32 = torch.cat([x_31, x_30, x_29, x_28, x_27, x_26, x_25, x_24, x_23], 1)
        x_32 = self.seq32(x_32)
        x_33 = torch.cat([x_32, x_31, x_30, x_29, x_28, x_27, x_26, x_25, x_24, x_23], 1)
        x_33 = self.seq33(x_33)
        x_34 = torch.cat([x_33, x_32, x_31, x_30, x_29, x_28, x_27, x_26, x_25, x_24, x_23], 1)
        x_34 = self.seq34(x_34)
        x_35 = torch.cat([x_34, x_33, x_32, x_31, x_30, x_29, x_28, x_27, x_26, x_25, x_24, x_23], 1)
        x_35 = self.seq35(x_35)
        x_36 = torch.cat([x_35, x_34, x_33, x_32, x_31, x_30, x_29, x_28, x_27, x_26, x_25, x_24, x_23], 1)
        x_36 = self.seq36(x_36)
        x_37 = torch.cat([x_36, x_35, x_34, x_33, x_32, x_31, x_30, x_29, x_28, x_27, x_26, x_25, x_24, x_23], 1)
        x_37 = self.seq37(x_37)
        x_38 = torch.cat([x_37, x_36, x_35, x_34, x_33, x_32, x_31, x_30, x_29, x_28, x_27, x_26, x_25, x_24, x_23], 1)
        x_38 = self.seq38(x_38)
        x_39 = torch.cat([x_38, x_37, x_36, x_35, x_34, x_33, x_32, x_31, x_30, x_29, x_28, x_27, x_26, x_25, x_24, x_23], 1)
        x_39 = self.seq39(x_39)
        x_40 = torch.cat([x_39, x_38, x_37, x_36, x_35, x_34, x_33, x_32, x_31, x_30, x_29, x_28, x_27, x_26, x_25, x_24, x_23], 1)
        x_40 = self.seq40(x_40)
        x_41 = torch.cat([x_40, x_39, x_38, x_37, x_36, x_35, x_34, x_33, x_32, x_31, x_30, x_29, x_28, x_27, x_26, x_25, x_24, x_23], 1)
        x_41 = self.seq41(x_41)
        x_42 = torch.cat([x_41, x_40, x_39, x_38, x_37, x_36, x_35, x_34, x_33, x_32, x_31, x_30, x_29, x_28, x_27, x_26, x_25, x_24, x_23], 1)
        x_42 = self.seq42(x_42)
        x_43 = torch.cat([x_42, x_41, x_40, x_39, x_38, x_37, x_36, x_35, x_34, x_33, x_32, x_31, x_30, x_29, x_28, x_27, x_26, x_25, x_24, x_23], 1)
        x_43 = self.seq43(x_43)
        x_44 = torch.cat([x_43, x_42, x_41, x_40, x_39, x_38, x_37, x_36, x_35, x_34, x_33, x_32, x_31, x_30, x_29, x_28, x_27, x_26, x_25, x_24, x_23], 1)
        x_44 = self.seq44(x_44)
        x_45 = torch.cat([x_44, x_43, x_42, x_41, x_40, x_39, x_38, x_37, x_36, x_35, x_34, x_33, x_32, x_31, x_30, x_29, x_28, x_27, x_26, x_25, x_24, x_23], 1)
        x_45 = self.seq45(x_45)
        x_46 = torch.cat([x_45, x_44, x_43, x_42, x_41, x_40, x_39, x_38, x_37, x_36, x_35, x_34, x_33, x_32, x_31, x_30, x_29, x_28, x_27, x_26, x_25, x_24, x_23], 1)
        x_46 = self.seq46(x_46)
        x_47 = torch.cat([x_46, x_45, x_44, x_43, x_42, x_41, x_40, x_39, x_38, x_37, x_36, x_35, x_34, x_33, x_32, x_31, x_30, x_29, x_28, x_27, x_26, x_25, x_24, x_23], 1)
        x_47 = self.seq47(x_47)
        x_48 = torch.cat([x_47, x_46, x_45, x_44, x_43, x_42, x_41, x_40, x_39, x_38, x_37, x_36, x_35, x_34, x_33, x_32, x_31, x_30, x_29, x_28, x_27, x_26, x_25, x_24, x_23], 1)
        x_48 = self.seq48(x_48)
        x_49 = torch.cat([x_48, x_47, x_46, x_45, x_44, x_43, x_42, x_41, x_40, x_39, x_38, x_37, x_36, x_35, x_34, x_33, x_32, x_31, x_30, x_29, x_28, x_27, x_26, x_25, x_24, x_23], 1)
        x_49 = self.seq49(x_49)
        x_50 = torch.cat([x_49, x_48, x_47, x_46, x_45, x_44, x_43, x_42, x_41, x_40, x_39, x_38, x_37, x_36, x_35, x_34, x_33, x_32, x_31, x_30, x_29, x_28, x_27, x_26, x_25, x_24, x_23], 1)
        x_50 = self.seq50(x_50)
        x_51 = torch.cat([x_50, x_49, x_48, x_47, x_46, x_45, x_44, x_43, x_42, x_41, x_40, x_39, x_38, x_37, x_36, x_35, x_34, x_33, x_32, x_31, x_30, x_29, x_28, x_27, x_26, x_25, x_24, x_23], 1)
        x_51 = self.seq51(x_51)
        x_52 = torch.cat([x_51, x_50, x_49, x_48, x_47, x_46, x_45, x_44, x_43, x_42, x_41, x_40, x_39, x_38, x_37, x_36, x_35, x_34, x_33, x_32, x_31, x_30, x_29, x_28, x_27, x_26, x_25, x_24, x_23], 1)
        x_52 = self.seq52(x_52)
        x_53 = torch.cat([x_52, x_51, x_50, x_49, x_48, x_47, x_46, x_45, x_44, x_43, x_42, x_41, x_40, x_39, x_38, x_37, x_36, x_35, x_34, x_33, x_32, x_31, x_30, x_29, x_28, x_27, x_26, x_25, x_24, x_23], 1)
        x_53 = self.seq53(x_53)
        x_54 = torch.cat([x_53, x_52, x_51, x_50, x_49, x_48, x_47, x_46, x_45, x_44, x_43, x_42, x_41, x_40, x_39, x_38, x_37, x_36, x_35, x_34, x_33, x_32, x_31, x_30, x_29, x_28, x_27, x_26, x_25, x_24, x_23], 1)
        x_54 = self.seq54(x_54)
        x_55 = torch.cat([x_54, x_53, x_52, x_51, x_50, x_49, x_48, x_47, x_46, x_45, x_44, x_43, x_42, x_41, x_40, x_39, x_38, x_37, x_36, x_35, x_34, x_33, x_32, x_31, x_30, x_29, x_28, x_27, x_26, x_25, x_24, x_23], 1)
        x_55 = self.seq55(x_55)
        x_56 = torch.cat([x_55, x_54, x_53, x_52, x_51, x_50, x_49, x_48, x_47, x_46, x_45, x_44, x_43, x_42, x_41, x_40, x_39, x_38, x_37, x_36, x_35, x_34, x_33, x_32, x_31, x_30, x_29, x_28, x_27, x_26, x_25, x_24, x_23], 1)
        x_56 = self.seq56(x_56)
        x_57 = torch.cat([x_56, x_55, x_54, x_53, x_52, x_51, x_50, x_49, x_48, x_47, x_46, x_45, x_44, x_43, x_42, x_41, x_40, x_39, x_38, x_37, x_36, x_35, x_34, x_33, x_32, x_31, x_30, x_29, x_28, x_27, x_26, x_25, x_24, x_23], 1)
        x_57 = self.seq57(x_57)
        x_58 = torch.cat([x_57, x_56, x_55, x_54, x_53, x_52, x_51, x_50, x_49, x_48, x_47, x_46, x_45, x_44, x_43, x_42, x_41, x_40, x_39, x_38, x_37, x_36, x_35, x_34, x_33, x_32, x_31, x_30, x_29, x_28, x_27, x_26, x_25, x_24, x_23], 1)
        x_58 = self.seq58(x_58)
        x_59 = torch.cat([x_58, x_57, x_56, x_55, x_54, x_53, x_52, x_51, x_50, x_49, x_48, x_47, x_46, x_45, x_44, x_43, x_42, x_41, x_40, x_39, x_38, x_37, x_36, x_35, x_34, x_33, x_32, x_31, x_30, x_29, x_28, x_27, x_26, x_25, x_24, x_23], 1)
        x_59 = self.seq59(x_59)
        x_60 = torch.cat([x_59, x_58, x_57, x_56, x_55, x_54, x_53, x_52, x_51, x_50, x_49, x_48, x_47, x_46, x_45, x_44, x_43, x_42, x_41, x_40, x_39, x_38, x_37, x_36, x_35, x_34, x_33, x_32, x_31, x_30, x_29, x_28, x_27, x_26, x_25, x_24, x_23], 1)
        x_60 = self.seq60(x_60)
        x_61 = torch.cat([x_60, x_59, x_58, x_57, x_56, x_55, x_54, x_53, x_52, x_51, x_50, x_49, x_48, x_47, x_46, x_45, x_44, x_43, x_42, x_41, x_40, x_39, x_38, x_37, x_36, x_35, x_34, x_33, x_32, x_31, x_30, x_29, x_28, x_27, x_26, x_25, x_24, x_23], 1)
        x_61 = self.seq61(x_61)
        x_62 = torch.cat([x_61, x_60, x_59, x_58, x_57, x_56, x_55, x_54, x_53, x_52, x_51, x_50, x_49, x_48, x_47, x_46, x_45, x_44, x_43, x_42, x_41, x_40, x_39, x_38, x_37, x_36, x_35, x_34, x_33, x_32, x_31, x_30, x_29, x_28, x_27, x_26, x_25, x_24, x_23], 1)
        x_62 = self.seq62(x_62)
        x_63 = torch.cat([x_62, x_61, x_60, x_59, x_58, x_57, x_56, x_55, x_54, x_53, x_52, x_51, x_50, x_49, x_48, x_47, x_46, x_45, x_44, x_43, x_42, x_41, x_40, x_39, x_38, x_37, x_36, x_35, x_34, x_33, x_32, x_31, x_30, x_29, x_28, x_27, x_26, x_25, x_24, x_23], 1)
        x_63 = self.seq63(x_63)
        x_64 = torch.cat([x_63, x_62, x_61, x_60, x_59, x_58, x_57, x_56, x_55, x_54, x_53, x_52, x_51, x_50, x_49, x_48, x_47, x_46, x_45, x_44, x_43, x_42, x_41, x_40, x_39, x_38, x_37, x_36, x_35, x_34, x_33, x_32, x_31, x_30, x_29, x_28, x_27, x_26, x_25, x_24, x_23], 1)
        x_64 = self.seq64(x_64)
        x_65 = torch.cat([x_64, x_63, x_62, x_61, x_60, x_59, x_58, x_57, x_56, x_55, x_54, x_53, x_52, x_51, x_50, x_49, x_48, x_47, x_46, x_45, x_44, x_43, x_42, x_41, x_40, x_39, x_38, x_37, x_36, x_35, x_34, x_33, x_32, x_31, x_30, x_29, x_28, x_27, x_26, x_25, x_24, x_23], 1)
        x_65 = self.seq65(x_65)
        x_66 = torch.cat([x_65, x_64, x_63, x_62, x_61, x_60, x_59, x_58, x_57, x_56, x_55, x_54, x_53, x_52, x_51, x_50, x_49, x_48, x_47, x_46, x_45, x_44, x_43, x_42, x_41, x_40, x_39, x_38, x_37, x_36, x_35, x_34, x_33, x_32, x_31, x_30, x_29, x_28, x_27, x_26, x_25, x_24, x_23], 1)
        x_66 = self.seq66(x_66)
        x_67 = torch.cat([x_66, x_65, x_64, x_63, x_62, x_61, x_60, x_59, x_58, x_57, x_56, x_55, x_54, x_53, x_52, x_51, x_50, x_49, x_48, x_47, x_46, x_45, x_44, x_43, x_42, x_41, x_40, x_39, x_38, x_37, x_36, x_35, x_34, x_33, x_32, x_31, x_30, x_29, x_28, x_27, x_26, x_25, x_24, x_23], 1)
        x_67 = self.seq67(x_67)
        x_68 = torch.cat([x_67, x_66, x_65, x_64, x_63, x_62, x_61, x_60, x_59, x_58, x_57, x_56, x_55, x_54, x_53, x_52, x_51, x_50, x_49, x_48, x_47, x_46, x_45, x_44, x_43, x_42, x_41, x_40, x_39, x_38, x_37, x_36, x_35, x_34, x_33, x_32, x_31, x_30, x_29, x_28, x_27, x_26, x_25, x_24, x_23], 1)
        x_68 = self.seq68(x_68)
        x_69 = torch.cat([x_68, x_67, x_66, x_65, x_64, x_63, x_62, x_61, x_60, x_59, x_58, x_57, x_56, x_55, x_54, x_53, x_52, x_51, x_50, x_49, x_48, x_47, x_46, x_45, x_44, x_43, x_42, x_41, x_40, x_39, x_38, x_37, x_36, x_35, x_34, x_33, x_32, x_31, x_30, x_29, x_28, x_27, x_26, x_25, x_24, x_23], 1)
        x_69 = self.seq69(x_69)
        x_70 = torch.cat([x_69, x_68, x_67, x_66, x_65, x_64, x_63, x_62, x_61, x_60, x_59, x_58, x_57, x_56, x_55, x_54, x_53, x_52, x_51, x_50, x_49, x_48, x_47, x_46, x_45, x_44, x_43, x_42, x_41, x_40, x_39, x_38, x_37, x_36, x_35, x_34, x_33, x_32, x_31, x_30, x_29, x_28, x_27, x_26, x_25, x_24, x_23], 1)
        x_70 = self.seq70(x_70)
        x_71 = torch.cat([x_70, x_69, x_68, x_67, x_66, x_65, x_64, x_63, x_62, x_61, x_60, x_59, x_58, x_57, x_56, x_55, x_54, x_53, x_52, x_51, x_50, x_49, x_48, x_47, x_46, x_45, x_44, x_43, x_42, x_41, x_40, x_39, x_38, x_37, x_36, x_35, x_34, x_33, x_32, x_31, x_30, x_29, x_28, x_27, x_26, x_25, x_24, x_23], 1)
        x_71 = self.seq71(x_71)
        x_72 = torch.cat([x_71, x_70, x_69, x_68, x_67, x_66, x_65, x_64, x_63, x_62, x_61, x_60, x_59, x_58, x_57, x_56, x_55, x_54, x_53, x_52, x_51, x_50, x_49, x_48, x_47, x_46, x_45, x_44, x_43, x_42, x_41, x_40, x_39, x_38, x_37, x_36, x_35, x_34, x_33, x_32, x_31, x_30, x_29, x_28, x_27, x_26, x_25, x_24, x_23], 1)
        x_72 = self.seq72(x_72)
        x_73 = self.seq73(x_72)
        x_74 = torch.cat([x_73], 1)
        x_74 = self.seq74(x_74)
        x_75 = torch.cat([x_74, x_73], 1)
        x_75 = self.seq75(x_75)
        x_76 = torch.cat([x_75, x_74, x_73], 1)
        x_76 = self.seq76(x_76)
        x_77 = torch.cat([x_76, x_75, x_74, x_73], 1)
        x_77 = self.seq77(x_77)
        x_78 = torch.cat([x_77, x_76, x_75, x_74, x_73], 1)
        x_78 = self.seq78(x_78)
        x_79 = torch.cat([x_78, x_77, x_76, x_75, x_74, x_73], 1)
        x_79 = self.seq79(x_79)
        x_80 = torch.cat([x_79, x_78, x_77, x_76, x_75, x_74, x_73], 1)
        x_80 = self.seq80(x_80)
        x_81 = torch.cat([x_80, x_79, x_78, x_77, x_76, x_75, x_74, x_73], 1)
        x_81 = self.seq81(x_81)
        x_82 = torch.cat([x_81, x_80, x_79, x_78, x_77, x_76, x_75, x_74, x_73], 1)
        x_82 = self.seq82(x_82)
        x_83 = torch.cat([x_82, x_81, x_80, x_79, x_78, x_77, x_76, x_75, x_74, x_73], 1)
        x_83 = self.seq83(x_83)
        x_84 = torch.cat([x_83, x_82, x_81, x_80, x_79, x_78, x_77, x_76, x_75, x_74, x_73], 1)
        x_84 = self.seq84(x_84)
        x_85 = torch.cat([x_84, x_83, x_82, x_81, x_80, x_79, x_78, x_77, x_76, x_75, x_74, x_73], 1)
        x_85 = self.seq85(x_85)
        x_86 = torch.cat([x_85, x_84, x_83, x_82, x_81, x_80, x_79, x_78, x_77, x_76, x_75, x_74, x_73], 1)
        x_86 = self.seq86(x_86)
        x_87 = torch.cat([x_86, x_85, x_84, x_83, x_82, x_81, x_80, x_79, x_78, x_77, x_76, x_75, x_74, x_73], 1)
        x_87 = self.seq87(x_87)
        x_88 = torch.cat([x_87, x_86, x_85, x_84, x_83, x_82, x_81, x_80, x_79, x_78, x_77, x_76, x_75, x_74, x_73], 1)
        x_88 = self.seq88(x_88)
        x_89 = torch.cat([x_88, x_87, x_86, x_85, x_84, x_83, x_82, x_81, x_80, x_79, x_78, x_77, x_76, x_75, x_74, x_73], 1)
        x_89 = self.seq89(x_89)
        x_90 = torch.cat([x_89, x_88, x_87, x_86, x_85, x_84, x_83, x_82, x_81, x_80, x_79, x_78, x_77, x_76, x_75, x_74, x_73], 1)
        x_90 = self.seq90(x_90)
        x_91 = torch.cat([x_90, x_89, x_88, x_87, x_86, x_85, x_84, x_83, x_82, x_81, x_80, x_79, x_78, x_77, x_76, x_75, x_74, x_73], 1)
        x_91 = self.seq91(x_91)
        x_92 = torch.cat([x_91, x_90, x_89, x_88, x_87, x_86, x_85, x_84, x_83, x_82, x_81, x_80, x_79, x_78, x_77, x_76, x_75, x_74, x_73], 1)
        x_92 = self.seq92(x_92)
        x_93 = torch.cat([x_92, x_91, x_90, x_89, x_88, x_87, x_86, x_85, x_84, x_83, x_82, x_81, x_80, x_79, x_78, x_77, x_76, x_75, x_74, x_73], 1)
        x_93 = self.seq93(x_93)
        x_94 = torch.cat([x_93, x_92, x_91, x_90, x_89, x_88, x_87, x_86, x_85, x_84, x_83, x_82, x_81, x_80, x_79, x_78, x_77, x_76, x_75, x_74, x_73], 1)
        x_94 = self.seq94(x_94)
        x_95 = torch.cat([x_94, x_93, x_92, x_91, x_90, x_89, x_88, x_87, x_86, x_85, x_84, x_83, x_82, x_81, x_80, x_79, x_78, x_77, x_76, x_75, x_74, x_73], 1)
        x_95 = self.seq95(x_95)
        x_96 = torch.cat([x_95, x_94, x_93, x_92, x_91, x_90, x_89, x_88, x_87, x_86, x_85, x_84, x_83, x_82, x_81, x_80, x_79, x_78, x_77, x_76, x_75, x_74, x_73], 1)
        x_96 = self.seq96(x_96)
        x_97 = torch.cat([x_96, x_95, x_94, x_93, x_92, x_91, x_90, x_89, x_88, x_87, x_86, x_85, x_84, x_83, x_82, x_81, x_80, x_79, x_78, x_77, x_76, x_75, x_74, x_73], 1)
        x_97 = self.seq97(x_97)
        x_98 = torch.cat([x_97, x_96, x_95, x_94, x_93, x_92, x_91, x_90, x_89, x_88, x_87, x_86, x_85, x_84, x_83, x_82, x_81, x_80, x_79, x_78, x_77, x_76, x_75, x_74, x_73], 1)
        x_98 = self.seq98(x_98)
        x_99 = torch.cat([x_98, x_97, x_96, x_95, x_94, x_93, x_92, x_91, x_90, x_89, x_88, x_87, x_86, x_85, x_84, x_83, x_82, x_81, x_80, x_79, x_78, x_77, x_76, x_75, x_74, x_73], 1)
        x_99 = self.seq99(x_99)
        x_100 = torch.cat([x_99, x_98, x_97, x_96, x_95, x_94, x_93, x_92, x_91, x_90, x_89, x_88, x_87, x_86, x_85, x_84, x_83, x_82, x_81, x_80, x_79, x_78, x_77, x_76, x_75, x_74, x_73], 1)
        x_100 = self.seq100(x_100)
        x_101 = torch.cat([x_100, x_99, x_98, x_97, x_96, x_95, x_94, x_93, x_92, x_91, x_90, x_89, x_88, x_87, x_86, x_85, x_84, x_83, x_82, x_81, x_80, x_79, x_78, x_77, x_76, x_75, x_74, x_73], 1)
        x_101 = self.seq101(x_101)
        x_102 = torch.cat([x_101, x_100, x_99, x_98, x_97, x_96, x_95, x_94, x_93, x_92, x_91, x_90, x_89, x_88, x_87, x_86, x_85, x_84, x_83, x_82, x_81, x_80, x_79, x_78, x_77, x_76, x_75, x_74, x_73], 1)
        x_102 = self.seq102(x_102)
        x_103 = torch.cat([x_102, x_101, x_100, x_99, x_98, x_97, x_96, x_95, x_94, x_93, x_92, x_91, x_90, x_89, x_88, x_87, x_86, x_85, x_84, x_83, x_82, x_81, x_80, x_79, x_78, x_77, x_76, x_75, x_74, x_73], 1)
        x_103 = self.seq103(x_103)
        x_104 = torch.cat([x_103, x_102, x_101, x_100, x_99, x_98, x_97, x_96, x_95, x_94, x_93, x_92, x_91, x_90, x_89, x_88, x_87, x_86, x_85, x_84, x_83, x_82, x_81, x_80, x_79, x_78, x_77, x_76, x_75, x_74, x_73], 1)
        x_104 = self.seq104(x_104)
        x_105 = torch.cat([x_104, x_103, x_102, x_101, x_100, x_99, x_98, x_97, x_96, x_95, x_94, x_93, x_92, x_91, x_90, x_89, x_88, x_87, x_86, x_85, x_84, x_83, x_82, x_81, x_80, x_79, x_78, x_77, x_76, x_75, x_74, x_73], 1)
        x_105 = self.seq105(x_105)
        x_106 = torch.cat([x_105, x_104, x_103, x_102, x_101, x_100, x_99, x_98, x_97, x_96, x_95, x_94, x_93, x_92, x_91, x_90, x_89, x_88, x_87, x_86, x_85, x_84, x_83, x_82, x_81, x_80, x_79, x_78, x_77, x_76, x_75, x_74, x_73], 1)
        x_106 = self.seq106(x_106)
        return x_106
