import torch
from torch import nn


class ConvertedResNet(nn.Module):

    def __init__(self):
        super(ConvertedResNet, self).__init__()
        self.seq1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=64, eps=1e-05),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding=[1, 1]),
        )
        self.seq2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=64, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=64, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
        )
        self.seq3 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=64, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=64, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
        )
        self.seq5 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq6 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=64, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=64, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
        )
        self.seq7 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq8 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
        )
        self.seq9 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq10 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
        )
        self.seq11 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq12 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
        )
        self.seq13 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq14 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
        )
        self.seq15 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq16 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq17 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq18 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq19 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq20 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq21 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq22 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq23 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq24 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq25 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq26 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq27 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq28 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq29 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq30 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq31 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq32 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq33 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq34 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq35 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq36 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq37 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq38 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq39 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq40 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq41 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq42 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq43 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq44 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq45 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq46 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq47 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq48 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq49 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq50 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq51 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq52 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq53 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq54 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq55 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq56 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq57 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq58 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq59 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq60 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq61 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq62 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=2048, eps=1e-05),
        )
        self.seq63 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq64 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=2048, eps=1e-05),
        )
        self.seq65 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq66 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=2048, eps=1e-05),
        )
        self.seq67 = nn.Sequential(
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=2048, out_features=1000),
        )
        self.seq68 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=(1, 1), stride=(2, 2), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=2048, eps=1e-05),
        )
        self.seq69 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(1, 1), stride=(2, 2), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-05),
        )
        self.seq70 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(1, 1), stride=(2, 2), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
        )
        self.seq71 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
        )

    def forward(self, x_0):
        x_1 = self.seq1(x_0)
        x_71 = self.seq71(x_1)
        x_2 = self.seq2(x_1)
        x_3 = x_2 + x_71
        x_3 = self.seq3(x_3)
        x_4 = self.seq4(x_3)
        x_5 = x_4 + x_3
        x_5 = self.seq5(x_5)
        x_6 = self.seq6(x_5)
        x_7 = x_6 + x_5
        x_7 = self.seq7(x_7)
        x_70 = self.seq70(x_7)
        x_8 = self.seq8(x_7)
        x_9 = x_8 + x_70
        x_9 = self.seq9(x_9)
        x_10 = self.seq10(x_9)
        x_11 = x_10 + x_9
        x_11 = self.seq11(x_11)
        x_12 = self.seq12(x_11)
        x_13 = x_12 + x_11
        x_13 = self.seq13(x_13)
        x_14 = self.seq14(x_13)
        x_15 = x_14 + x_13
        x_15 = self.seq15(x_15)
        x_69 = self.seq69(x_15)
        x_16 = self.seq16(x_15)
        x_17 = x_16 + x_69
        x_17 = self.seq17(x_17)
        x_18 = self.seq18(x_17)
        x_19 = x_18 + x_17
        x_19 = self.seq19(x_19)
        x_20 = self.seq20(x_19)
        x_21 = x_20 + x_19
        x_21 = self.seq21(x_21)
        x_22 = self.seq22(x_21)
        x_23 = x_22 + x_21
        x_23 = self.seq23(x_23)
        x_24 = self.seq24(x_23)
        x_25 = x_24 + x_23
        x_25 = self.seq25(x_25)
        x_26 = self.seq26(x_25)
        x_27 = x_26 + x_25
        x_27 = self.seq27(x_27)
        x_28 = self.seq28(x_27)
        x_29 = x_28 + x_27
        x_29 = self.seq29(x_29)
        x_30 = self.seq30(x_29)
        x_31 = x_30 + x_29
        x_31 = self.seq31(x_31)
        x_32 = self.seq32(x_31)
        x_33 = x_32 + x_31
        x_33 = self.seq33(x_33)
        x_34 = self.seq34(x_33)
        x_35 = x_34 + x_33
        x_35 = self.seq35(x_35)
        x_36 = self.seq36(x_35)
        x_37 = x_36 + x_35
        x_37 = self.seq37(x_37)
        x_38 = self.seq38(x_37)
        x_39 = x_38 + x_37
        x_39 = self.seq39(x_39)
        x_40 = self.seq40(x_39)
        x_41 = x_40 + x_39
        x_41 = self.seq41(x_41)
        x_42 = self.seq42(x_41)
        x_43 = x_42 + x_41
        x_43 = self.seq43(x_43)
        x_44 = self.seq44(x_43)
        x_45 = x_44 + x_43
        x_45 = self.seq45(x_45)
        x_46 = self.seq46(x_45)
        x_47 = x_46 + x_45
        x_47 = self.seq47(x_47)
        x_48 = self.seq48(x_47)
        x_49 = x_48 + x_47
        x_49 = self.seq49(x_49)
        x_50 = self.seq50(x_49)
        x_51 = x_50 + x_49
        x_51 = self.seq51(x_51)
        x_52 = self.seq52(x_51)
        x_53 = x_52 + x_51
        x_53 = self.seq53(x_53)
        x_54 = self.seq54(x_53)
        x_55 = x_54 + x_53
        x_55 = self.seq55(x_55)
        x_56 = self.seq56(x_55)
        x_57 = x_56 + x_55
        x_57 = self.seq57(x_57)
        x_58 = self.seq58(x_57)
        x_59 = x_58 + x_57
        x_59 = self.seq59(x_59)
        x_60 = self.seq60(x_59)
        x_61 = x_60 + x_59
        x_61 = self.seq61(x_61)
        x_68 = self.seq68(x_61)
        x_62 = self.seq62(x_61)
        x_63 = x_62 + x_68
        x_63 = self.seq63(x_63)
        x_64 = self.seq64(x_63)
        x_65 = x_64 + x_63
        x_65 = self.seq65(x_65)
        x_66 = self.seq66(x_65)
        x_67 = x_66 + x_65
        x_67 = self.seq67(x_67)
        return x_67
