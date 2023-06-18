import torch
from torch import nn


class NaiveConvertResnet(nn.Module):

    def __init__(self, in_shape=3):
        super(NaiveConvertResnet, self).__init__()
        self.seq1 = nn.Sequential(
            nn.Conv2d(in_channels=in_shape, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=[3, 3], dilation=(1, 1)),
            nn.BatchNorm2d(num_features=64, eps=1e-05),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding=[1, 1]),
        )
        self.seq2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=[1, 1], dilation=(1, 1)),
            nn.BatchNorm2d(num_features=64, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=[1, 1], dilation=(1, 1)),
            nn.BatchNorm2d(num_features=64, eps=1e-05),
        )
        self.seq3 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=[1, 1], dilation=(1, 1)),
            nn.BatchNorm2d(num_features=64, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=[1, 1], dilation=(1, 1)),
            nn.BatchNorm2d(num_features=64, eps=1e-05),
        )
        self.seq5 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq6 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(2, 2), padding=[1, 1], dilation=(1, 1)),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=[1, 1], dilation=(1, 1)),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
        )
        self.seq7 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq8 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=[1, 1], dilation=(1, 1)),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=[1, 1], dilation=(1, 1)),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
        )
        self.seq9 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq10 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(2, 2), padding=[1, 1], dilation=(1, 1)),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=[1, 1], dilation=(1, 1)),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
        )
        self.seq11 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq12 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=[1, 1], dilation=(1, 1)),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=[1, 1], dilation=(1, 1)),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
        )
        self.seq13 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq14 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(2, 2), padding=[1, 1], dilation=(1, 1)),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=[1, 1], dilation=(1, 1)),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
        )
        self.seq15 = nn.Sequential(
            nn.ReLU(),
        )
        self.seq16 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=[1, 1], dilation=(1, 1)),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=[1, 1], dilation=(1, 1)),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
        )
        self.seq17 = nn.Sequential(
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=512, out_features=10),
        )
        self.seq18 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(1, 1), stride=(2, 2), padding=[0, 0], dilation=(1, 1)),
            nn.BatchNorm2d(num_features=512, eps=1e-05),
        )
        self.seq19 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), stride=(2, 2), padding=[0, 0], dilation=(1, 1)),
            nn.BatchNorm2d(num_features=256, eps=1e-05),
        )
        self.seq20 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), stride=(2, 2), padding=[0, 0], dilation=(1, 1)),
            nn.BatchNorm2d(num_features=128, eps=1e-05),
        )

    def forward(self, x_0):
        x_1 = self.seq1(x_0)
        x_2 = self.seq2(x_1)
        x_3 = x_2 + x_1
        x_3 = self.seq3(x_3)
        x_4 = self.seq4(x_3)
        x_5 = x_4 + x_3
        x_5 = self.seq5(x_5)
        x_20 = self.seq20(x_5)
        x_6 = self.seq6(x_5)
        x_7 = x_6 + x_20
        x_7 = self.seq7(x_7)
        x_8 = self.seq8(x_7)
        x_9 = x_8 + x_7
        x_9 = self.seq9(x_9)
        x_19 = self.seq19(x_9)
        x_10 = self.seq10(x_9)
        x_11 = x_10 + x_19
        x_11 = self.seq11(x_11)
        x_12 = self.seq12(x_11)
        x_13 = x_12 + x_11
        x_13 = self.seq13(x_13)
        x_18 = self.seq18(x_13)
        x_14 = self.seq14(x_13)
        x_15 = x_14 + x_18
        x_15 = self.seq15(x_15)
        x_16 = self.seq16(x_15)
        x_17 = x_16 + x_15
        x_17 = self.seq17(x_17)
        return x_17
