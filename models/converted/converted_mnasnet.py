import torch
from torch import nn


class ConvertedMnasNet(nn.Module):

    def __init__(self, in_shape=3):
        super(ConvertedMnasNet, self).__init__()
        self.seq1 = nn.Sequential(
            nn.Conv2d(in_channels=in_shape, out_channels=40, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=40, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=40, out_channels=40, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=40),
            nn.BatchNorm2d(num_features=40, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=40, out_channels=24, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=24, eps=1e-05),
            nn.Conv2d(in_channels=24, out_channels=72, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=72, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=72, out_channels=72, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), dilation=(1, 1), groups=72),
            nn.BatchNorm2d(num_features=72, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=72, out_channels=32, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=32, eps=1e-05),
        )
        self.seq2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=96, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=96, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=96),
            nn.BatchNorm2d(num_features=96, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=96, out_channels=32, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=32, eps=1e-05),
        )
        self.seq4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=96, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=96, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=96),
            nn.BatchNorm2d(num_features=96, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=96, out_channels=32, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=32, eps=1e-05),
        )
        self.seq5 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=96, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=96, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), dilation=(1, 1), groups=96),
            nn.BatchNorm2d(num_features=96, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=96, out_channels=56, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=56, eps=1e-05),
        )
        self.seq6 = nn.Sequential(
            nn.Conv2d(in_channels=56, out_channels=168, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=168, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=168),
            nn.BatchNorm2d(num_features=168, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=168, out_channels=56, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=56, eps=1e-05),
        )
        self.seq8 = nn.Sequential(
            nn.Conv2d(in_channels=56, out_channels=168, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=168, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=168, out_channels=168, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=168),
            nn.BatchNorm2d(num_features=168, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=168, out_channels=56, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=56, eps=1e-05),
        )
        self.seq9 = nn.Sequential(
            nn.Conv2d(in_channels=56, out_channels=336, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=336, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), dilation=(1, 1), groups=336),
            nn.BatchNorm2d(num_features=336, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=336, out_channels=104, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=104, eps=1e-05),
        )
        self.seq10 = nn.Sequential(
            nn.Conv2d(in_channels=104, out_channels=624, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=624, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=624, out_channels=624, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=624),
            nn.BatchNorm2d(num_features=624, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=624, out_channels=104, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=104, eps=1e-05),
        )
        self.seq12 = nn.Sequential(
            nn.Conv2d(in_channels=104, out_channels=624, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=624, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=624, out_channels=624, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=624),
            nn.BatchNorm2d(num_features=624, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=624, out_channels=104, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=104, eps=1e-05),
        )
        self.seq13 = nn.Sequential(
            nn.Conv2d(in_channels=104, out_channels=624, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=624, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=624, out_channels=624, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=624),
            nn.BatchNorm2d(num_features=624, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=624, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
        )
        self.seq14 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=768, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=768, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=768, out_channels=768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=768),
            nn.BatchNorm2d(num_features=768, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=768, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
        )
        self.seq15 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=768, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=768, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=768, out_channels=768, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), dilation=(1, 1), groups=768),
            nn.BatchNorm2d(num_features=768, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=768, out_channels=248, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=248, eps=1e-05),
        )
        self.seq16 = nn.Sequential(
            nn.Conv2d(in_channels=248, out_channels=1488, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1488, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=1488, out_channels=1488, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=1488),
            nn.BatchNorm2d(num_features=1488, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=1488, out_channels=248, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=248, eps=1e-05),
        )
        self.seq18 = nn.Sequential(
            nn.Conv2d(in_channels=248, out_channels=1488, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1488, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=1488, out_channels=1488, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=1488),
            nn.BatchNorm2d(num_features=1488, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=1488, out_channels=248, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=248, eps=1e-05),
        )
        self.seq20 = nn.Sequential(
            nn.Conv2d(in_channels=248, out_channels=1488, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1488, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=1488, out_channels=1488, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=1488),
            nn.BatchNorm2d(num_features=1488, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=1488, out_channels=248, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=248, eps=1e-05),
        )
        self.seq21 = nn.Sequential(
            nn.Conv2d(in_channels=248, out_channels=1488, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1488, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=1488, out_channels=1488, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1488),
            nn.BatchNorm2d(num_features=1488, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=1488, out_channels=416, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=416, eps=1e-05),
            nn.Conv2d(in_channels=416, out_channels=1280, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.BatchNorm2d(num_features=1280, eps=1e-05),
            nn.ReLU(),
        )
        self.seq22 = nn.Sequential(
            nn.Linear(in_features=1280, out_features=1000),
        )

    def forward(self, x_0):
        x_1 = self.seq1(x_0)
        x_2 = self.seq2(x_1)
        x_3 = x_2 + x_1
        x_4 = self.seq4(x_3)
        x_5 = x_4 + x_3
        x_5 = self.seq5(x_5)
        x_6 = self.seq6(x_5)
        x_7 = x_6 + x_5
        x_8 = self.seq8(x_7)
        x_9 = x_8 + x_7
        x_9 = self.seq9(x_9)
        x_10 = self.seq10(x_9)
        x_11 = x_10 + x_9
        x_12 = self.seq12(x_11)
        x_13 = x_12 + x_11
        x_13 = self.seq13(x_13)
        x_14 = self.seq14(x_13)
        x_15 = x_14 + x_13
        x_15 = self.seq15(x_15)
        x_16 = self.seq16(x_15)
        x_17 = x_16 + x_15
        x_18 = self.seq18(x_17)
        x_19 = x_18 + x_17
        x_20 = self.seq20(x_19)
        x_21 = x_20 + x_19
        x_21 = self.seq21(x_21)
        x_22 = x_21.mean([2, 3])
        x_22 = self.seq22(x_22)
        return x_22
