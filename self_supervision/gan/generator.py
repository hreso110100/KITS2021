from torch import nn, cat
from torch.nn import Conv2d, Sequential, Dropout, ConvTranspose2d, LeakyReLU, ReLU, InstanceNorm2d, Tanh, MaxPool2d


class UNetDown(nn.Module):
    def __init__(self, input_size: int, output_filters: int, dropout=0.0, pooling=True, is_first=False):
        super(UNetDown, self).__init__()
        self.model = Sequential()

        if pooling:
            self.model.add_module("MaxPooling2d", MaxPool2d(kernel_size=2, stride=2))

        if is_first:
            self.model.add_module("Conv2d", Conv2d(input_size, output_filters, kernel_size=(1, 3), padding=(0, 1)))
            self.model.add_module("InstanceNorm2d", InstanceNorm2d(output_filters))
            self.model.add_module("LeakyReLU", LeakyReLU(0.2, inplace=True))
        else:
            self.model.add_module("Conv2d", Conv2d(input_size, output_filters, kernel_size=3, padding=1))
            self.model.add_module("InstanceNorm2d", InstanceNorm2d(output_filters))
            self.model.add_module("LeakyReLU", LeakyReLU(0.2, inplace=True))

        if is_first:
            self.model.add_module("Conv2dSecond",
                                  Conv2d(output_filters, output_filters, kernel_size=(1, 3), padding=(0, 1)))
            self.model.add_module("InstanceNorm2dSecond", InstanceNorm2d(output_filters))
            self.model.add_module("LeakyReLUSecond", LeakyReLU(0.2, inplace=True))
        else:
            self.model.add_module("Conv2dSecond",
                                  Conv2d(output_filters, output_filters, kernel_size=3, padding=1))
            self.model.add_module("InstanceNorm2dSecond", InstanceNorm2d(output_filters))
            self.model.add_module("LeakyReLUSecond", LeakyReLU(0.2, inplace=True))

        if dropout:
            self.model.add_module("Dropout", Dropout(dropout))

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, input_size: int, output_filters: int, dropout=0.0, is_diff_than_conv=False):
        super(UNetUp, self).__init__()

        self.conv = Sequential(
            Conv2d(input_size, output_filters, kernel_size=3, padding=1, bias=True),
            InstanceNorm2d(output_filters),
            ReLU(inplace=True),
            Conv2d(output_filters, output_filters, kernel_size=3, padding=1, bias=True),
            InstanceNorm2d(output_filters),
            ReLU(inplace=True),
        )

        if dropout:
            self.conv.add_module("Dropout", Dropout(dropout))

        if is_diff_than_conv:
            self.transpose = ConvTranspose2d(320, output_filters, kernel_size=2, stride=2, bias=False)
        else:
            self.transpose = ConvTranspose2d(input_size, output_filters, kernel_size=2, stride=2, bias=False)

    def forward(self, layer, skip_input):
        layer = self.transpose(layer)
        concatenated = cat((layer, skip_input), 1)
        return self.conv(concatenated)


"""
Implementation based on UNet generator
"""


class Generator(nn.Module):
    def __init__(self, file_shape: tuple):
        super(Generator, self).__init__()

        # DownSampling
        self.down1 = UNetDown(file_shape[0], 32, pooling=False, is_first=True)
        self.down2 = UNetDown(32, 64)
        self.down3 = UNetDown(64, 128)
        self.down4 = UNetDown(128, 256)
        self.down5 = UNetDown(256, 320)
        self.down6 = UNetDown(320, 320)
        self.down7 = UNetDown(320, 320)

        # UpSampling
        self.up1 = UNetUp(640, 320, is_diff_than_conv=True, dropout=0.1)
        self.up2 = UNetUp(640, 320, is_diff_than_conv=True, dropout=0.1)
        self.up3 = UNetUp(512, 256, is_diff_than_conv=True, dropout=0.1)
        self.up4 = UNetUp(256, 128, dropout=0.1)
        self.up5 = UNetUp(128, 64, dropout=0.1)
        self.up6 = UNetUp(64, 32, dropout=0.1)

        self.last = nn.Sequential(
            ConvTranspose2d(32, file_shape[0], kernel_size=3, stride=1, padding=1, dilation=1),
            Tanh(),
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)

        u1 = self.up1(d7, d6)
        u2 = self.up2(u1, d5)
        u3 = self.up3(u2, d4)
        u4 = self.up4(u3, d3)
        u5 = self.up5(u4, d2)
        u6 = self.up6(u5, d1)

        return self.last(u6)
