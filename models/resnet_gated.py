import torch
import torch.nn as nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential, BatchNorm2d, ReLU, AdaptiveAvgPool2d

class FrequencyResnetGated(nn.Module):
    def __init__(self, in_channels=96, out_channel=96):
        super(FrequencyResnetGated, self).__init__()
        self.model0 = Sequential(
        Conv2d(in_channels=in_channels * 3, out_channels=64, kernel_size=(7, 7), stride=1, padding=1),
        BatchNorm2d(64),
        ReLU(),
        MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),
        )
        self.model1 = Sequential(
        Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
        BatchNorm2d(64),
        ReLU(),
        Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
        BatchNorm2d(64),
        ReLU(),
        )
        self.R1 = ReLU()
        self.model2 = Sequential(
        Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
        BatchNorm2d(64),
        ReLU(),
        Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
        BatchNorm2d(64),
        ReLU(),
        )
        self.R2 = ReLU()
        self.model3 = Sequential(
        Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
        BatchNorm2d(128),
        ReLU(),
        Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
        BatchNorm2d(128),
        ReLU(),
        )
        self.en1 = Sequential(
        Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), stride=1, padding=0),
        BatchNorm2d(128),
        ReLU(),
        )
        self.R3 = ReLU()
        self.model4 = Sequential(
        Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
        BatchNorm2d(128),
        ReLU(),
        Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
        BatchNorm2d(128),
        ReLU(),
        )
        self.R4 = ReLU()
        self.model5 = Sequential(
        Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=2, padding=1),
        BatchNorm2d(256),
        ReLU(),
        Conv2d(in_channels=256, out_channels=out_channel, kernel_size=(3, 3), stride=1, padding=1),
        BatchNorm2d(out_channel),
        ReLU(),
        )
        self.en2 = Sequential(
        Conv2d(in_channels=128, out_channels=out_channel, kernel_size=(1, 1), stride=2, padding=0),
        BatchNorm2d(out_channel),
        ReLU(),
        )
        self.R5 = ReLU()
        # self.model6 = Sequential(
        # Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
        # BatchNorm2d(256),
        # ReLU(),
        # Conv2d(in_channels=256, out_channels=out_channel, kernel_size=(3, 3), stride=1, padding=1),
        # BatchNorm2d(256),
        # ReLU(),
        # )
        # self.R6 = ReLU()
        self.gate = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1)
        # self.model7 = Sequential(
        # Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=2, padding=1),
        # BatchNorm2d(512),
        # ReLU(),
        # Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
        # BatchNorm2d(512),
        # ReLU(),
        # )
        # self.en3 = Sequential(
        # Conv2d(in_channels=256, out_channels=512, kernel_size=(1, 1), stride=2, padding=0),
        # BatchNorm2d(512),
        # ReLU(),
        # )
        # self.R7 = ReLU()
        # self.model8 = Sequential(
        # Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
        # BatchNorm2d(512),
        # ReLU(),
        # Conv2d(in_channels=512, out_channels=out_channel, kernel_size=(3, 3), stride=1, padding=1),
        # BatchNorm2d(out_channel),
        # ReLU(),
        # )
        # self.R8 = ReLU()
        # self.aap = AdaptiveAvgPool2d((1, 1))
        # self.flatten = Flatten(start_dim=1)
        # self.fc = Linear(512, num_classes)

    def forward(self, x):
        x = self.model0(x)
        f1 = x
        x = self.model1(x)
        x = x + f1
        x = self.R1(x)
        f1_1 = x
        x = self.model2(x)
        x = x + f1_1
        x = self.R2(x)
        f2_1 = x
        f2_1 = self.en1(f2_1)
        x = self.model3(x)
        x = x + f2_1
        x = self.R3(x)
        f2_2 = x
        x = self.model4(x)
        x = x + f2_2
        x = self.R4(x)
        f3_1 = x
        f3_1 = self.en2(f3_1)
        x = self.model5(x)
        x = x + f3_1
        x = self.R5(x)
        # f3_2 = x
        # x = self.model6(x)
        # x = x + f3_2

        # x = self.R6(x)
        gate = torch.sigmoid(self.gate(x))
        x = x * gate

        return x