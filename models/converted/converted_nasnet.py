import torch
from torch import nn


class ConvertedNasNet(nn.Module):

    def __init__(self):
        super(ConvertedNasNet, self).__init__()
        self.seq1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=96, eps=0.001),
        )
        self.seq2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=96, out_channels=42, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=42, eps=0.001),
        )
        self.seq3 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=42, out_channels=42, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), dilation=(1, 1), groups=42),
            nn.Conv2d(in_channels=42, out_channels=42, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=42, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=42, out_channels=42, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=42),
            nn.Conv2d(in_channels=42, out_channels=42, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=42, eps=0.001),
        )
        self.seq5 = nn.Sequential(
            nn.AdaptiveAvgPool2d((83, 83)),
        )
        self.seq8 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=168, out_channels=84, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=84, eps=0.001),
        )
        self.seq9 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=84, out_channels=84, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), dilation=(1, 1), groups=84),
            nn.Conv2d(in_channels=84, out_channels=84, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=84, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=84, out_channels=84, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=84),
            nn.Conv2d(in_channels=84, out_channels=84, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=84, eps=0.001),
        )
        self.seq11 = nn.Sequential(
            nn.AdaptiveAvgPool2d((42, 42)),
        )
        self.seq14 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=336, out_channels=168, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=168, eps=0.001),
        )
        self.seq15 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=168),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=168, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=168),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=168, eps=0.001),
        )
        self.seq18 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=1008, out_channels=168, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=168, eps=0.001),
        )
        self.seq19 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=168),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=168, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=168),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=168, eps=0.001),
        )
        self.seq22 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=1008, out_channels=168, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=168, eps=0.001),
        )
        self.seq23 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=168),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=168, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=168),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=168, eps=0.001),
        )
        self.seq26 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=1008, out_channels=168, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=168, eps=0.001),
        )
        self.seq27 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=168),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=168, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=168),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=168, eps=0.001),
        )
        self.seq30 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=1008, out_channels=168, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=168, eps=0.001),
        )
        self.seq31 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=168),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=168, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=168),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=168, eps=0.001),
        )
        self.seq34 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=1008, out_channels=168, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=168, eps=0.001),
        )
        self.seq35 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=168),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=168, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=168),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=168, eps=0.001),
        )
        self.seq37 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=1008, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
        )
        self.seq38 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq39 = nn.Sequential(
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), dilation=(1, 1), groups=336),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=336),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
        )
        self.seq41 = nn.Sequential(
            nn.AdaptiveAvgPool2d((21, 21)),
        )
        self.seq44 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=1344, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
        )
        self.seq45 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=336),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=336),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
        )
        self.seq48 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=2016, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
        )
        self.seq49 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=336),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=336),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
        )
        self.seq52 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=2016, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
        )
        self.seq53 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=336),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=336),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
        )
        self.seq56 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=2016, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
        )
        self.seq57 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=336),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=336),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
        )
        self.seq60 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=2016, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
        )
        self.seq61 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=336),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=336),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
        )
        self.seq64 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=2016, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
        )
        self.seq65 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=336),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=336),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
        )
        self.seq67 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=2016, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
        )
        self.seq68 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), dilation=(1, 1), groups=672),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=672),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
        )
        self.seq70 = nn.Sequential(
            nn.AdaptiveAvgPool2d((11, 11)),
        )
        self.seq73 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=2688, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
        )
        self.seq74 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=672),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=672),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
        )
        self.seq77 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=4032, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
        )
        self.seq78 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=672),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=672),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
        )
        self.seq81 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=4032, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
        )
        self.seq82 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=672),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=672),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
        )
        self.seq85 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=4032, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
        )
        self.seq86 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=672),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=672),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
        )
        self.seq89 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=4032, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
        )
        self.seq90 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=672),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=672),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
        )
        self.seq92 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=4032, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
        )
        self.seq93 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=672),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=672),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
        )
        self.seq95 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq96 = nn.Sequential(
            nn.AvgPool2d(kernel_size=[11, 11], stride=[1, 1], padding=[0, 0]),
            nn.Flatten(),
            nn.Linear(in_features=4032, out_features=1000),
        )
        self.seq97 = nn.Sequential(
            nn.AdaptiveAvgPool2d((11, 11)),
        )
        self.seq99 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=672),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=672),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
        )
        self.seq101 = nn.Sequential(
            nn.AdaptiveAvgPool2d((11, 11)),
        )
        self.seq103 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=672),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=672),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
        )
        self.seq105 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=4032, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
        )
        self.seq106 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=672),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=672),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
        )
        self.seq107 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=672),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=672),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
        )
        self.seq109 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=672),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=672),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
        )
        self.seq110 = nn.Sequential(
            nn.AdaptiveAvgPool2d((11, 11)),
        )
        self.seq112 = nn.Sequential(
            nn.AdaptiveAvgPool2d((11, 11)),
        )
        self.seq113 = nn.Sequential(
            nn.AdaptiveAvgPool2d((11, 11)),
        )
        self.seq115 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=672),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=672),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
        )
        self.seq117 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=4032, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
        )
        self.seq118 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=672),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=672),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
        )
        self.seq119 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=672),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=672),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
        )
        self.seq121 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=672),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=672),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
        )
        self.seq122 = nn.Sequential(
            nn.AdaptiveAvgPool2d((11, 11)),
        )
        self.seq124 = nn.Sequential(
            nn.AdaptiveAvgPool2d((11, 11)),
        )
        self.seq125 = nn.Sequential(
            nn.AdaptiveAvgPool2d((11, 11)),
        )
        self.seq127 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=672),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=672),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
        )
        self.seq129 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=4032, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
        )
        self.seq130 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=672),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=672),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
        )
        self.seq131 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=672),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=672),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
        )
        self.seq133 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=672),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=672),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
        )
        self.seq134 = nn.Sequential(
            nn.AdaptiveAvgPool2d((11, 11)),
        )
        self.seq136 = nn.Sequential(
            nn.AdaptiveAvgPool2d((11, 11)),
        )
        self.seq137 = nn.Sequential(
            nn.AdaptiveAvgPool2d((11, 11)),
        )
        self.seq139 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=672),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=672),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
        )
        self.seq141 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=4032, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
        )
        self.seq142 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=672),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=672),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
        )
        self.seq143 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=672),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=672),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
        )
        self.seq145 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=672),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=672),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
        )
        self.seq146 = nn.Sequential(
            nn.AdaptiveAvgPool2d((11, 11)),
        )
        self.seq148 = nn.Sequential(
            nn.AdaptiveAvgPool2d((11, 11)),
        )
        self.seq149 = nn.Sequential(
            nn.AdaptiveAvgPool2d((11, 11)),
        )
        self.seq151 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=672),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=672),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
        )
        self.seq153 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=2688, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
        )
        self.seq154 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=672),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=672),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
        )
        self.seq155 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=672),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=672),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
        )
        self.seq157 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=672),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=672),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
        )
        self.seq158 = nn.Sequential(
            nn.AdaptiveAvgPool2d((11, 11)),
        )
        self.seq160 = nn.Sequential(
            nn.AdaptiveAvgPool2d((11, 11)),
        )
        self.seq161 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=672),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=672),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
        )
        self.seq163 = nn.Sequential(
            nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding=[1, 1]),
        )
        self.seq165 = nn.Sequential(
            nn.AdaptiveAvgPool2d((11, 11)),
        )
        self.seq167 = nn.Sequential(
            nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding=[1, 1]),
        )
        self.seq168 = nn.Sequential(
            nn.AdaptiveAvgPool2d((21, 21)),
        )
        self.seq170 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=336),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=336),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
        )
        self.seq172 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=2016, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
        )
        self.seq173 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), dilation=(1, 1), groups=672),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), dilation=(1, 1), groups=672),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
        )
        self.seq174 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), dilation=(1, 1), groups=672),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), dilation=(1, 1), groups=672),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
        )
        self.seq175 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), dilation=(1, 1), groups=672),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=672),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
        )
        self.seq176 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq177 = nn.Sequential(
            nn.AdaptiveAvgPool2d((11, 11)),
            nn.Conv2d(in_channels=2016, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
        )
        self.seq178 = nn.Sequential(
            nn.BatchNorm2d(num_features=672, eps=0.001),
        )
        self.seq179 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=672),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=672),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
        )
        self.seq180 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=672),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=672),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
        )
        self.seq182 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=672),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=672),
            nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=672, eps=0.001),
        )
        self.seq183 = nn.Sequential(
            nn.AdaptiveAvgPool2d((11, 11)),
        )
        self.seq185 = nn.Sequential(
            nn.AdaptiveAvgPool2d((11, 11)),
        )
        self.seq186 = nn.Sequential(
            nn.AdaptiveAvgPool2d((11, 11)),
            nn.Conv2d(in_channels=2016, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
        )
        self.seq187 = nn.Sequential(
            nn.AdaptiveAvgPool2d((21, 21)),
        )
        self.seq189 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=336),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=336),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
        )
        self.seq191 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=2016, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
        )
        self.seq192 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=336),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=336),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
        )
        self.seq193 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=336),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=336),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
        )
        self.seq195 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=336),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=336),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
        )
        self.seq196 = nn.Sequential(
            nn.AdaptiveAvgPool2d((21, 21)),
        )
        self.seq198 = nn.Sequential(
            nn.AdaptiveAvgPool2d((21, 21)),
        )
        self.seq199 = nn.Sequential(
            nn.AdaptiveAvgPool2d((21, 21)),
        )
        self.seq201 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=336),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=336),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
        )
        self.seq203 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=2016, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
        )
        self.seq204 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=336),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=336),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
        )
        self.seq205 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=336),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=336),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
        )
        self.seq207 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=336),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=336),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
        )
        self.seq208 = nn.Sequential(
            nn.AdaptiveAvgPool2d((21, 21)),
        )
        self.seq210 = nn.Sequential(
            nn.AdaptiveAvgPool2d((21, 21)),
        )
        self.seq211 = nn.Sequential(
            nn.AdaptiveAvgPool2d((21, 21)),
        )
        self.seq213 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=336),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=336),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
        )
        self.seq215 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=2016, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
        )
        self.seq216 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=336),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=336),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
        )
        self.seq217 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=336),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=336),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
        )
        self.seq219 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=336),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=336),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
        )
        self.seq220 = nn.Sequential(
            nn.AdaptiveAvgPool2d((21, 21)),
        )
        self.seq222 = nn.Sequential(
            nn.AdaptiveAvgPool2d((21, 21)),
        )
        self.seq223 = nn.Sequential(
            nn.AdaptiveAvgPool2d((21, 21)),
        )
        self.seq225 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=336),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=336),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
        )
        self.seq227 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=2016, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
        )
        self.seq228 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=336),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=336),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
        )
        self.seq229 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=336),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=336),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
        )
        self.seq231 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=336),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=336),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
        )
        self.seq232 = nn.Sequential(
            nn.AdaptiveAvgPool2d((21, 21)),
        )
        self.seq234 = nn.Sequential(
            nn.AdaptiveAvgPool2d((21, 21)),
        )
        self.seq235 = nn.Sequential(
            nn.AdaptiveAvgPool2d((21, 21)),
        )
        self.seq237 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=336),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=336),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
        )
        self.seq239 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=1344, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
        )
        self.seq240 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=336),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=336),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
        )
        self.seq241 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=336),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=336),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
        )
        self.seq243 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=336),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=336),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
        )
        self.seq244 = nn.Sequential(
            nn.AdaptiveAvgPool2d((21, 21)),
        )
        self.seq246 = nn.Sequential(
            nn.AdaptiveAvgPool2d((21, 21)),
        )
        self.seq247 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq248 = nn.Sequential(
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=336),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=336),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
        )
        self.seq250 = nn.Sequential(
            nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding=[1, 1]),
        )
        self.seq252 = nn.Sequential(
            nn.AvgPool2d(kernel_size=[3, 3], stride=[2, 2], padding=[1, 1]),
        )
        self.seq254 = nn.Sequential(
            nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding=[1, 1]),
        )
        self.seq255 = nn.Sequential(
            nn.AdaptiveAvgPool2d((42, 42)),
        )
        self.seq257 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=168),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=168, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=168),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=168, eps=0.001),
        )
        self.seq259 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=1008, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
        )
        self.seq260 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq261 = nn.Sequential(
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), dilation=(1, 1), groups=336),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), dilation=(1, 1), groups=336),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
        )
        self.seq262 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq263 = nn.Sequential(
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), dilation=(1, 1), groups=336),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), dilation=(1, 1), groups=336),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
        )
        self.seq264 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq265 = nn.Sequential(
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), dilation=(1, 1), groups=336),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=336),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
        )
        self.seq266 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq267 = nn.Sequential(
            nn.AdaptiveAvgPool2d((21, 21)),
            nn.Conv2d(in_channels=1008, out_channels=168, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
        )
        self.seq268 = nn.Sequential(
            nn.BatchNorm2d(num_features=336, eps=0.001),
        )
        self.seq269 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=336),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=336),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
        )
        self.seq270 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=336),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=336),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
        )
        self.seq272 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=336),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=336),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=0.001),
        )
        self.seq273 = nn.Sequential(
            nn.AdaptiveAvgPool2d((21, 21)),
        )
        self.seq275 = nn.Sequential(
            nn.AdaptiveAvgPool2d((21, 21)),
        )
        self.seq276 = nn.Sequential(
            nn.AdaptiveAvgPool2d((21, 21)),
            nn.Conv2d(in_channels=1008, out_channels=168, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
        )
        self.seq277 = nn.Sequential(
            nn.AdaptiveAvgPool2d((42, 42)),
        )
        self.seq279 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=168),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=168, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=168),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=168, eps=0.001),
        )
        self.seq281 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=1008, out_channels=168, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=168, eps=0.001),
        )
        self.seq282 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=168),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=168, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=168),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=168, eps=0.001),
        )
        self.seq283 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=168),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=168, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=168),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=168, eps=0.001),
        )
        self.seq285 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=168),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=168, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=168),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=168, eps=0.001),
        )
        self.seq286 = nn.Sequential(
            nn.AdaptiveAvgPool2d((42, 42)),
        )
        self.seq288 = nn.Sequential(
            nn.AdaptiveAvgPool2d((42, 42)),
        )
        self.seq289 = nn.Sequential(
            nn.AdaptiveAvgPool2d((42, 42)),
        )
        self.seq291 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=168),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=168, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=168),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=168, eps=0.001),
        )
        self.seq293 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=1008, out_channels=168, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=168, eps=0.001),
        )
        self.seq294 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=168),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=168, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=168),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=168, eps=0.001),
        )
        self.seq295 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=168),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=168, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=168),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=168, eps=0.001),
        )
        self.seq297 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=168),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=168, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=168),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=168, eps=0.001),
        )
        self.seq298 = nn.Sequential(
            nn.AdaptiveAvgPool2d((42, 42)),
        )
        self.seq300 = nn.Sequential(
            nn.AdaptiveAvgPool2d((42, 42)),
        )
        self.seq301 = nn.Sequential(
            nn.AdaptiveAvgPool2d((42, 42)),
        )
        self.seq303 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=168),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=168, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=168),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=168, eps=0.001),
        )
        self.seq305 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=1008, out_channels=168, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=168, eps=0.001),
        )
        self.seq306 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=168),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=168, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=168),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=168, eps=0.001),
        )
        self.seq307 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=168),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=168, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=168),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=168, eps=0.001),
        )
        self.seq309 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=168),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=168, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=168),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=168, eps=0.001),
        )
        self.seq310 = nn.Sequential(
            nn.AdaptiveAvgPool2d((42, 42)),
        )
        self.seq312 = nn.Sequential(
            nn.AdaptiveAvgPool2d((42, 42)),
        )
        self.seq313 = nn.Sequential(
            nn.AdaptiveAvgPool2d((42, 42)),
        )
        self.seq315 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=168),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=168, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=168),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=168, eps=0.001),
        )
        self.seq317 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=1008, out_channels=168, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=168, eps=0.001),
        )
        self.seq318 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=168),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=168, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=168),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=168, eps=0.001),
        )
        self.seq319 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=168),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=168, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=168),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=168, eps=0.001),
        )
        self.seq321 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=168),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=168, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=168),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=168, eps=0.001),
        )
        self.seq322 = nn.Sequential(
            nn.AdaptiveAvgPool2d((42, 42)),
        )
        self.seq324 = nn.Sequential(
            nn.AdaptiveAvgPool2d((42, 42)),
        )
        self.seq325 = nn.Sequential(
            nn.AdaptiveAvgPool2d((42, 42)),
        )
        self.seq327 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=168),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=168, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=168),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=168, eps=0.001),
        )
        self.seq329 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=336, out_channels=168, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=168, eps=0.001),
        )
        self.seq330 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=168),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=168, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=168),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=168, eps=0.001),
        )
        self.seq331 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=168),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=168, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=168),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=168, eps=0.001),
        )
        self.seq333 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=168),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=168, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=168),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=168, eps=0.001),
        )
        self.seq334 = nn.Sequential(
            nn.AdaptiveAvgPool2d((42, 42)),
        )
        self.seq336 = nn.Sequential(
            nn.AdaptiveAvgPool2d((42, 42)),
        )
        self.seq337 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=84, out_channels=84, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=84),
            nn.Conv2d(in_channels=84, out_channels=84, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=84, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=84, out_channels=84, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=84),
            nn.Conv2d(in_channels=84, out_channels=84, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=84, eps=0.001),
        )
        self.seq339 = nn.Sequential(
            nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding=[1, 1]),
        )
        self.seq341 = nn.Sequential(
            nn.AdaptiveAvgPool2d((42, 42)),
        )
        self.seq343 = nn.Sequential(
            nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding=[1, 1]),
        )
        self.seq344 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq345 = nn.Sequential(
            nn.AdaptiveAvgPool2d((42, 42)),
            nn.Conv2d(in_channels=168, out_channels=84, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
        )
        self.seq346 = nn.Sequential(
            nn.BatchNorm2d(num_features=168, eps=0.001),
        )
        self.seq347 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=168),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=168, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=168),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=168, eps=0.001),
        )
        self.seq348 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=168),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=168, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=168),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=168, eps=0.001),
        )
        self.seq350 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=168),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=168, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=168),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=168, eps=0.001),
        )
        self.seq351 = nn.Sequential(
            nn.AdaptiveAvgPool2d((42, 42)),
        )
        self.seq353 = nn.Sequential(
            nn.AdaptiveAvgPool2d((42, 42)),
        )
        self.seq354 = nn.Sequential(
            nn.AdaptiveAvgPool2d((42, 42)),
            nn.Conv2d(in_channels=168, out_channels=84, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
        )
        self.seq355 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=42, out_channels=42, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=42),
            nn.Conv2d(in_channels=42, out_channels=42, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=42, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=42, out_channels=42, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=42),
            nn.Conv2d(in_channels=42, out_channels=42, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=42, eps=0.001),
        )
        self.seq357 = nn.Sequential(
            nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding=[1, 1]),
        )
        self.seq359 = nn.Sequential(
            nn.AdaptiveAvgPool2d((83, 83)),
        )
        self.seq361 = nn.Sequential(
            nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding=[1, 1]),
        )
        self.seq362 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), dilation=(1, 1), groups=96),
            nn.Conv2d(in_channels=96, out_channels=42, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=42, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=42, out_channels=42, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), dilation=(1, 1), groups=42),
            nn.Conv2d(in_channels=42, out_channels=42, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=42, eps=0.001),
        )
        self.seq363 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), dilation=(1, 1), groups=96),
            nn.Conv2d(in_channels=96, out_channels=42, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=42, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=42, out_channels=42, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), dilation=(1, 1), groups=42),
            nn.Conv2d(in_channels=42, out_channels=42, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=42, eps=0.001),
        )
        self.seq364 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), dilation=(1, 1), groups=96),
            nn.Conv2d(in_channels=96, out_channels=42, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=42, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=42, out_channels=42, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=42),
            nn.Conv2d(in_channels=42, out_channels=42, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=42, eps=0.001),
        )
        self.seq365 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq366 = nn.Sequential(
            nn.AdaptiveAvgPool2d((83, 83)),
            nn.Conv2d(in_channels=96, out_channels=42, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
        )
        self.seq367 = nn.Sequential(
            nn.BatchNorm2d(num_features=84, eps=0.001),
        )
        self.seq368 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=84, out_channels=84, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), dilation=(1, 1), groups=84),
            nn.Conv2d(in_channels=84, out_channels=84, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=84, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=84, out_channels=84, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), dilation=(1, 1), groups=84),
            nn.Conv2d(in_channels=84, out_channels=84, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=84, eps=0.001),
        )
        self.seq369 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=84, out_channels=84, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), dilation=(1, 1), groups=84),
            nn.Conv2d(in_channels=84, out_channels=84, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=84, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=84, out_channels=84, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), dilation=(1, 1), groups=84),
            nn.Conv2d(in_channels=84, out_channels=84, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=84, eps=0.001),
        )
        self.seq370 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=84, out_channels=84, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), dilation=(1, 1), groups=84),
            nn.Conv2d(in_channels=84, out_channels=84, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=84, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=84, out_channels=84, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=84),
            nn.Conv2d(in_channels=84, out_channels=84, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=84, eps=0.001),
        )
        self.seq371 = nn.Sequential(
            nn.AdaptiveAvgPool2d((83, 83)),
            nn.Conv2d(in_channels=96, out_channels=42, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
        )

    def forward(self, x_0):
        x_1 = self.seq1(x_0)
        x_365 = self.seq365(x_1)
        x_371 = torch.nn.functional.pad(x_365, [0, 1, 0, 1, 0, 0, 0, 0])
        x_371 = self.seq371(x_371)
        x_366 = self.seq366(x_365)
        x_367 = torch.cat([x_366, x_371], 1)
        x_367 = self.seq367(x_367)
        x_370 = self.seq370(x_367)
        x_369 = self.seq369(x_367)
        x_368 = self.seq368(x_367)
        x_364 = self.seq364(x_1)
        x_363 = self.seq363(x_1)
        x_362 = self.seq362(x_1)
        x_2 = self.seq2(x_1)
        x_361 = self.seq361(x_2)
        x_359 = self.seq359(x_2)
        x_360 = x_359 + x_364
        x_357 = self.seq357(x_2)
        x_358 = x_357 + x_363
        x_3 = self.seq3(x_2)
        x_4 = x_3 + x_362
        x_355 = self.seq355(x_4)
        x_356 = x_355 + x_361
        x_5 = self.seq5(x_4)
        x_6 = x_5 + x_358
        x_7 = torch.cat([x_6, x_356, x_358, x_360], 1)
        x_344 = self.seq344(x_7)
        x_354 = torch.nn.functional.pad(x_344, [0, 1, 0, 1, 0, 0, 0, 0])
        x_354 = self.seq354(x_354)
        x_345 = self.seq345(x_344)
        x_346 = torch.cat([x_345, x_354], 1)
        x_346 = self.seq346(x_346)
        x_353 = self.seq353(x_346)
        x_351 = self.seq351(x_346)
        x_352 = x_351 + x_353
        x_350 = self.seq350(x_346)
        x_348 = self.seq348(x_346)
        x_349 = x_348 + x_350
        x_347 = self.seq347(x_346)
        x_8 = self.seq8(x_7)
        x_343 = self.seq343(x_8)
        x_341 = self.seq341(x_8)
        x_342 = x_341 + x_370
        x_339 = self.seq339(x_8)
        x_340 = x_339 + x_369
        x_9 = self.seq9(x_8)
        x_10 = x_9 + x_368
        x_337 = self.seq337(x_10)
        x_338 = x_337 + x_343
        x_11 = self.seq11(x_10)
        x_12 = x_11 + x_340
        x_13 = torch.cat([x_12, x_338, x_340, x_342], 1)
        x_329 = self.seq329(x_13)
        x_336 = self.seq336(x_329)
        x_334 = self.seq334(x_329)
        x_335 = x_334 + x_336
        x_333 = self.seq333(x_329)
        x_331 = self.seq331(x_329)
        x_332 = x_331 + x_333
        x_330 = self.seq330(x_329)
        x_14 = self.seq14(x_13)
        x_327 = self.seq327(x_14)
        x_328 = x_327 + x_14
        x_325 = self.seq325(x_14)
        x_326 = x_325 + x_346
        x_15 = self.seq15(x_14)
        x_16 = x_15 + x_347
        x_17 = torch.cat([x_16, x_326, x_328, x_349, x_352, x_346], 1)
        x_317 = self.seq317(x_17)
        x_324 = self.seq324(x_317)
        x_322 = self.seq322(x_317)
        x_323 = x_322 + x_324
        x_321 = self.seq321(x_317)
        x_319 = self.seq319(x_317)
        x_320 = x_319 + x_321
        x_318 = self.seq318(x_317)
        x_18 = self.seq18(x_17)
        x_315 = self.seq315(x_18)
        x_316 = x_315 + x_18
        x_313 = self.seq313(x_18)
        x_314 = x_313 + x_329
        x_19 = self.seq19(x_18)
        x_20 = x_19 + x_330
        x_21 = torch.cat([x_20, x_314, x_316, x_332, x_335, x_329], 1)
        x_305 = self.seq305(x_21)
        x_312 = self.seq312(x_305)
        x_310 = self.seq310(x_305)
        x_311 = x_310 + x_312
        x_309 = self.seq309(x_305)
        x_307 = self.seq307(x_305)
        x_308 = x_307 + x_309
        x_306 = self.seq306(x_305)
        x_22 = self.seq22(x_21)
        x_303 = self.seq303(x_22)
        x_304 = x_303 + x_22
        x_301 = self.seq301(x_22)
        x_302 = x_301 + x_317
        x_23 = self.seq23(x_22)
        x_24 = x_23 + x_318
        x_25 = torch.cat([x_24, x_302, x_304, x_320, x_323, x_317], 1)
        x_293 = self.seq293(x_25)
        x_300 = self.seq300(x_293)
        x_298 = self.seq298(x_293)
        x_299 = x_298 + x_300
        x_297 = self.seq297(x_293)
        x_295 = self.seq295(x_293)
        x_296 = x_295 + x_297
        x_294 = self.seq294(x_293)
        x_26 = self.seq26(x_25)
        x_291 = self.seq291(x_26)
        x_292 = x_291 + x_26
        x_289 = self.seq289(x_26)
        x_290 = x_289 + x_305
        x_27 = self.seq27(x_26)
        x_28 = x_27 + x_306
        x_29 = torch.cat([x_28, x_290, x_292, x_308, x_311, x_305], 1)
        x_281 = self.seq281(x_29)
        x_288 = self.seq288(x_281)
        x_286 = self.seq286(x_281)
        x_287 = x_286 + x_288
        x_285 = self.seq285(x_281)
        x_283 = self.seq283(x_281)
        x_284 = x_283 + x_285
        x_282 = self.seq282(x_281)
        x_30 = self.seq30(x_29)
        x_279 = self.seq279(x_30)
        x_280 = x_279 + x_30
        x_277 = self.seq277(x_30)
        x_278 = x_277 + x_293
        x_31 = self.seq31(x_30)
        x_32 = x_31 + x_294
        x_33 = torch.cat([x_32, x_278, x_280, x_296, x_299, x_293], 1)
        x_266 = self.seq266(x_33)
        x_276 = torch.nn.functional.pad(x_266, [0, 1, 0, 1, 0, 0, 0, 0])
        x_276 = self.seq276(x_276)
        x_267 = self.seq267(x_266)
        x_268 = torch.cat([x_267, x_276], 1)
        x_268 = self.seq268(x_268)
        x_275 = self.seq275(x_268)
        x_273 = self.seq273(x_268)
        x_274 = x_273 + x_275
        x_272 = self.seq272(x_268)
        x_270 = self.seq270(x_268)
        x_271 = x_270 + x_272
        x_269 = self.seq269(x_268)
        x_259 = self.seq259(x_33)
        x_264 = self.seq264(x_259)
        x_265 = torch.nn.functional.pad(x_264, [1, 0, 1, 0, 0, 0, 0, 0])
        x_265 = self.seq265(x_265)
        x_262 = self.seq262(x_259)
        x_263 = torch.nn.functional.pad(x_262, [1, 0, 1, 0, 0, 0, 0, 0])
        x_263 = self.seq263(x_263)
        x_260 = self.seq260(x_259)
        x_261 = torch.nn.functional.pad(x_260, [1, 0, 1, 0, 0, 0, 0, 0])
        x_261 = self.seq261(x_261)
        x_34 = self.seq34(x_33)
        x_257 = self.seq257(x_34)
        x_258 = x_257 + x_34
        x_255 = self.seq255(x_34)
        x_256 = x_255 + x_281
        x_35 = self.seq35(x_34)
        x_36 = x_35 + x_282
        x_37 = torch.cat([x_36, x_256, x_258, x_284, x_287, x_281], 1)
        x_37 = self.seq37(x_37)
        x_254 = torch.nn.functional.pad(x_37, [1, 0, 1, 0, 0, 0, 0, 0])
        x_254 = self.seq254(x_254)
        x_252 = torch.nn.functional.pad(x_37, [1, 0, 1, 0, 0, 0, 0, 0])
        x_252 = self.seq252(x_252)
        x_253 = x_252 + x_265
        x_250 = torch.nn.functional.pad(x_37, [1, 0, 1, 0, 0, 0, 0, 0])
        x_250 = self.seq250(x_250)
        x_251 = x_250 + x_263
        x_38 = self.seq38(x_37)
        x_39 = torch.nn.functional.pad(x_38, [1, 0, 1, 0, 0, 0, 0, 0])
        x_39 = self.seq39(x_39)
        x_40 = x_39 + x_261
        x_247 = self.seq247(x_40)
        x_248 = torch.nn.functional.pad(x_247, [1, 0, 1, 0, 0, 0, 0, 0])
        x_248 = self.seq248(x_248)
        x_249 = x_248
        x_41 = self.seq41(x_40)
        x_42 = x_41 + x_251
        x_43 = torch.cat([x_42, x_249, x_251, x_253], 1)
        x_239 = self.seq239(x_43)
        x_246 = self.seq246(x_239)
        x_244 = self.seq244(x_239)
        x_245 = x_244 + x_246
        x_243 = self.seq243(x_239)
        x_241 = self.seq241(x_239)
        x_242 = x_241 + x_243
        x_240 = self.seq240(x_239)
        x_44 = self.seq44(x_43)
        x_237 = self.seq237(x_44)
        x_238 = x_237 + x_44
        x_235 = self.seq235(x_44)
        x_236 = x_235 + x_268
        x_45 = self.seq45(x_44)
        x_46 = x_45 + x_269
        x_47 = torch.cat([x_46, x_236, x_238, x_271, x_274, x_268], 1)
        x_227 = self.seq227(x_47)
        x_234 = self.seq234(x_227)
        x_232 = self.seq232(x_227)
        x_233 = x_232 + x_234
        x_231 = self.seq231(x_227)
        x_229 = self.seq229(x_227)
        x_230 = x_229 + x_231
        x_228 = self.seq228(x_227)
        x_48 = self.seq48(x_47)
        x_225 = self.seq225(x_48)
        x_226 = x_225 + x_48
        x_223 = self.seq223(x_48)
        x_224 = x_223 + x_239
        x_49 = self.seq49(x_48)
        x_50 = x_49 + x_240
        x_51 = torch.cat([x_50, x_224, x_226, x_242, x_245, x_239], 1)
        x_215 = self.seq215(x_51)
        x_222 = self.seq222(x_215)
        x_220 = self.seq220(x_215)
        x_221 = x_220 + x_222
        x_219 = self.seq219(x_215)
        x_217 = self.seq217(x_215)
        x_218 = x_217 + x_219
        x_216 = self.seq216(x_215)
        x_52 = self.seq52(x_51)
        x_213 = self.seq213(x_52)
        x_214 = x_213 + x_52
        x_211 = self.seq211(x_52)
        x_212 = x_211 + x_227
        x_53 = self.seq53(x_52)
        x_54 = x_53 + x_228
        x_55 = torch.cat([x_54, x_212, x_214, x_230, x_233, x_227], 1)
        x_203 = self.seq203(x_55)
        x_210 = self.seq210(x_203)
        x_208 = self.seq208(x_203)
        x_209 = x_208 + x_210
        x_207 = self.seq207(x_203)
        x_205 = self.seq205(x_203)
        x_206 = x_205 + x_207
        x_204 = self.seq204(x_203)
        x_56 = self.seq56(x_55)
        x_201 = self.seq201(x_56)
        x_202 = x_201 + x_56
        x_199 = self.seq199(x_56)
        x_200 = x_199 + x_215
        x_57 = self.seq57(x_56)
        x_58 = x_57 + x_216
        x_59 = torch.cat([x_58, x_200, x_202, x_218, x_221, x_215], 1)
        x_191 = self.seq191(x_59)
        x_198 = self.seq198(x_191)
        x_196 = self.seq196(x_191)
        x_197 = x_196 + x_198
        x_195 = self.seq195(x_191)
        x_193 = self.seq193(x_191)
        x_194 = x_193 + x_195
        x_192 = self.seq192(x_191)
        x_60 = self.seq60(x_59)
        x_189 = self.seq189(x_60)
        x_190 = x_189 + x_60
        x_187 = self.seq187(x_60)
        x_188 = x_187 + x_203
        x_61 = self.seq61(x_60)
        x_62 = x_61 + x_204
        x_63 = torch.cat([x_62, x_188, x_190, x_206, x_209, x_203], 1)
        x_176 = self.seq176(x_63)
        x_186 = torch.nn.functional.pad(x_176, [0, 1, 0, 1, 0, 0, 0, 0])
        x_186 = self.seq186(x_186)
        x_177 = self.seq177(x_176)
        x_178 = torch.cat([x_177, x_186], 1)
        x_178 = self.seq178(x_178)
        x_185 = self.seq185(x_178)
        x_183 = self.seq183(x_178)
        x_184 = x_183 + x_185
        x_182 = self.seq182(x_178)
        x_180 = self.seq180(x_178)
        x_181 = x_180 + x_182
        x_179 = self.seq179(x_178)
        x_172 = self.seq172(x_63)
        x_175 = self.seq175(x_172)
        x_174 = self.seq174(x_172)
        x_173 = self.seq173(x_172)
        x_64 = self.seq64(x_63)
        x_170 = self.seq170(x_64)
        x_171 = x_170 + x_64
        x_168 = self.seq168(x_64)
        x_169 = x_168 + x_191
        x_65 = self.seq65(x_64)
        x_66 = x_65 + x_192
        x_67 = torch.cat([x_66, x_169, x_171, x_194, x_197, x_191], 1)
        x_67 = self.seq67(x_67)
        x_167 = self.seq167(x_67)
        x_165 = self.seq165(x_67)
        x_166 = x_165 + x_175
        x_163 = self.seq163(x_67)
        x_164 = x_163 + x_174
        x_68 = self.seq68(x_67)
        x_69 = x_68 + x_173
        x_161 = self.seq161(x_69)
        x_162 = x_161 + x_167
        x_70 = self.seq70(x_69)
        x_71 = x_70 + x_164
        x_72 = torch.cat([x_71, x_162, x_164, x_166], 1)
        x_153 = self.seq153(x_72)
        x_160 = self.seq160(x_153)
        x_158 = self.seq158(x_153)
        x_159 = x_158 + x_160
        x_157 = self.seq157(x_153)
        x_155 = self.seq155(x_153)
        x_156 = x_155 + x_157
        x_154 = self.seq154(x_153)
        x_73 = self.seq73(x_72)
        x_151 = self.seq151(x_73)
        x_152 = x_151 + x_73
        x_149 = self.seq149(x_73)
        x_150 = x_149 + x_178
        x_74 = self.seq74(x_73)
        x_75 = x_74 + x_179
        x_76 = torch.cat([x_75, x_150, x_152, x_181, x_184, x_178], 1)
        x_141 = self.seq141(x_76)
        x_148 = self.seq148(x_141)
        x_146 = self.seq146(x_141)
        x_147 = x_146 + x_148
        x_145 = self.seq145(x_141)
        x_143 = self.seq143(x_141)
        x_144 = x_143 + x_145
        x_142 = self.seq142(x_141)
        x_77 = self.seq77(x_76)
        x_139 = self.seq139(x_77)
        x_140 = x_139 + x_77
        x_137 = self.seq137(x_77)
        x_138 = x_137 + x_153
        x_78 = self.seq78(x_77)
        x_79 = x_78 + x_154
        x_80 = torch.cat([x_79, x_138, x_140, x_156, x_159, x_153], 1)
        x_129 = self.seq129(x_80)
        x_136 = self.seq136(x_129)
        x_134 = self.seq134(x_129)
        x_135 = x_134 + x_136
        x_133 = self.seq133(x_129)
        x_131 = self.seq131(x_129)
        x_132 = x_131 + x_133
        x_130 = self.seq130(x_129)
        x_81 = self.seq81(x_80)
        x_127 = self.seq127(x_81)
        x_128 = x_127 + x_81
        x_125 = self.seq125(x_81)
        x_126 = x_125 + x_141
        x_82 = self.seq82(x_81)
        x_83 = x_82 + x_142
        x_84 = torch.cat([x_83, x_126, x_128, x_144, x_147, x_141], 1)
        x_117 = self.seq117(x_84)
        x_124 = self.seq124(x_117)
        x_122 = self.seq122(x_117)
        x_123 = x_122 + x_124
        x_121 = self.seq121(x_117)
        x_119 = self.seq119(x_117)
        x_120 = x_119 + x_121
        x_118 = self.seq118(x_117)
        x_85 = self.seq85(x_84)
        x_115 = self.seq115(x_85)
        x_116 = x_115 + x_85
        x_113 = self.seq113(x_85)
        x_114 = x_113 + x_129
        x_86 = self.seq86(x_85)
        x_87 = x_86 + x_130
        x_88 = torch.cat([x_87, x_114, x_116, x_132, x_135, x_129], 1)
        x_105 = self.seq105(x_88)
        x_112 = self.seq112(x_105)
        x_110 = self.seq110(x_105)
        x_111 = x_110 + x_112
        x_109 = self.seq109(x_105)
        x_107 = self.seq107(x_105)
        x_108 = x_107 + x_109
        x_106 = self.seq106(x_105)
        x_89 = self.seq89(x_88)
        x_103 = self.seq103(x_89)
        x_104 = x_103 + x_89
        x_101 = self.seq101(x_89)
        x_102 = x_101 + x_117
        x_90 = self.seq90(x_89)
        x_91 = x_90 + x_118
        x_92 = torch.cat([x_91, x_102, x_104, x_120, x_123, x_117], 1)
        x_92 = self.seq92(x_92)
        x_99 = self.seq99(x_92)
        x_100 = x_99 + x_92
        x_97 = self.seq97(x_92)
        x_98 = x_97 + x_105
        x_93 = self.seq93(x_92)
        x_94 = x_93 + x_106
        x_95 = torch.cat([x_94, x_98, x_100, x_108, x_111, x_105], 1)
        x_95 = self.seq95(x_95)
        x_96 = self.seq96(x_95)
        return x_96
