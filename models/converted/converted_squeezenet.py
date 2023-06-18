import torch
from torch import nn


class ConvertedSqueezeNet(nn.Module):

    def __init__(self):
        super(ConvertedSqueezeNet, self).__init__()
        self.seq1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding=[0, 0], ceil_mode=True),
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
        )
        self.seq2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
        )
        self.seq3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=16, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
        )
        self.seq4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
        )
        self.seq5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding=[0, 0], ceil_mode=True),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
        )
        self.seq6 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
        )
        self.seq7 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=32, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
        )
        self.seq8 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
        )
        self.seq9 = nn.Sequential(
            nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding=[0, 0], ceil_mode=True),
            nn.Conv2d(in_channels=256, out_channels=48, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
        )
        self.seq10 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=192, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
        )
        self.seq11 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=48, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
        )
        self.seq12 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=192, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
        )
        self.seq13 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
        )
        self.seq14 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
        )
        self.seq15 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
        )
        self.seq16 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
        )
        self.seq17 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1000, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.seq18 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.ReLU(),
        )
        self.seq19 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.ReLU(),
        )
        self.seq20 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.ReLU(),
        )
        self.seq21 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.ReLU(),
        )
        self.seq22 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.ReLU(),
        )
        self.seq23 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.ReLU(),
        )
        self.seq24 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.ReLU(),
        )
        self.seq25 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.ReLU(),
        )

    def forward(self, x_0):
        x_1 = self.seq1(x_0)
        x_25 = self.seq25(x_1)
        x_2 = self.seq2(x_1)
        x_3 = torch.cat([x_2, x_25], 1)
        x_3 = self.seq3(x_3)
        x_24 = self.seq24(x_3)
        x_4 = self.seq4(x_3)
        x_5 = torch.cat([x_4, x_24], 1)
        x_5 = self.seq5(x_5)
        x_23 = self.seq23(x_5)
        x_6 = self.seq6(x_5)
        x_7 = torch.cat([x_6, x_23], 1)
        x_7 = self.seq7(x_7)
        x_22 = self.seq22(x_7)
        x_8 = self.seq8(x_7)
        x_9 = torch.cat([x_8, x_22], 1)
        x_9 = self.seq9(x_9)
        x_21 = self.seq21(x_9)
        x_10 = self.seq10(x_9)
        x_11 = torch.cat([x_10, x_21], 1)
        x_11 = self.seq11(x_11)
        x_20 = self.seq20(x_11)
        x_12 = self.seq12(x_11)
        x_13 = torch.cat([x_12, x_20], 1)
        x_13 = self.seq13(x_13)
        x_19 = self.seq19(x_13)
        x_14 = self.seq14(x_13)
        x_15 = torch.cat([x_14, x_19], 1)
        x_15 = self.seq15(x_15)
        x_18 = self.seq18(x_15)
        x_16 = self.seq16(x_15)
        x_17 = torch.cat([x_16, x_18], 1)
        x_17 = self.seq17(x_17)
        return x_17
