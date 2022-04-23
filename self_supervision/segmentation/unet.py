from torch import nn, cat
from torch.nn import Conv2d, Sequential, Dropout, ConvTranspose2d, LeakyReLU, ReLU, BatchNorm2d, MaxPool2d


class UNetDown(nn.Module):
    def __init__(self, input_size: int, output_filters: int, dropout=0.0, pooling=True):
        super(UNetDown, self).__init__()
        self.model = Sequential()

        if pooling:
            self.model.add_module("MaxPooling2d", MaxPool2d(kernel_size=2, stride=2))

        self.model.add_module("Conv2d", Conv2d(input_size, output_filters, kernel_size=3, padding=1, bias=False))
        self.model.add_module("BatchNorm2d", BatchNorm2d(output_filters))
        self.model.add_module("LeakyReLU", LeakyReLU(0.2, inplace=True))

        self.model.add_module("Conv2dSecond", Conv2d(output_filters, output_filters, kernel_size=3, padding=1, bias=False))
        self.model.add_module("BatchNorm2dSecond", BatchNorm2d(output_filters))
        self.model.add_module("LeakyReLUSecond", LeakyReLU(0.2, inplace=True))

        if dropout:
            self.model.add_module("Dropout", Dropout(dropout))

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, input_size: int, output_filters: int, dropout=0.0):
        super(UNetUp, self).__init__()

        self.conv = Sequential(
            Conv2d(input_size, output_filters, kernel_size=3, padding=1, bias=False),
            BatchNorm2d(output_filters),
            ReLU(inplace=True),
            Conv2d(output_filters, output_filters, kernel_size=3, padding=1, bias=False),
            BatchNorm2d(output_filters),
            ReLU(inplace=True),
        )

        if dropout:
            self.conv.add_module("Dropout", Dropout(dropout))

        self.transpose = ConvTranspose2d(input_size, output_filters, kernel_size=2, stride=2)

    def forward(self, layer, skip_input):
        layer = self.transpose(layer)
        concatenated = cat((layer, skip_input), 1)
        return self.conv(concatenated)


"""
Implementation based on UNet generator
"""


class UNet(nn.Module):
    def __init__(self, file_shape: tuple):
        super(UNet, self).__init__()

        # DownSampling
        self.down1 = UNetDown(file_shape[0], 64, pooling=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512)
        self.bottleneck = UNetDown(512, 1024)

        # UpSampling
        self.up1 = UNetUp(1024, 512)
        self.up2 = UNetUp(512, 256)
        self.up3 = UNetUp(256, 128)
        self.up4 = UNetUp(128, 64)

        self.output = nn.Sequential(
            Conv2d(64, 4, kernel_size=1),
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        bottleneck = self.bottleneck(d4)

        u1 = self.up1(bottleneck, d4)
        u2 = self.up2(u1, d3)
        u3 = self.up3(u2, d2)
        u4 = self.up4(u3, d1)

        return self.output(u4)
