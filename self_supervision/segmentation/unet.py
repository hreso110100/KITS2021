from torch import nn, cat
from torch.nn import Conv2d, Sequential, Dropout, ConvTranspose2d, LeakyReLU, ReLU, InstanceNorm2d


class UNetDown(nn.Module):
    def __init__(self, input_size: int, output_filters: int, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()

        self.model = Sequential(
            Conv2d(input_size, output_filters, kernel_size=4, stride=2, padding=1, bias=False),
        )
        if normalize:
            self.model.add_module("InstanceNorm2d", InstanceNorm2d(output_filters))

        self.model.add_module("LeakyReLU", LeakyReLU(0.2, inplace=True))

        if dropout:
            self.model.add_module("Dropout", Dropout(dropout))

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, input_size: int, output_filters: int, dropout=0.0):
        super(UNetUp, self).__init__()

        self.model = Sequential(
            ConvTranspose2d(input_size, output_filters, kernel_size=4, stride=2, padding=1, bias=False),
            InstanceNorm2d(output_filters),
            ReLU(inplace=True),
        )

        if dropout:
            self.model.add_module("Dropout", Dropout(dropout))

    def forward(self, layer, skip_input):
        layer = self.model(layer)
        layer = cat((layer, skip_input), 1)

        return layer


"""
Implementation based on UNet generator
"""


class UNet(nn.Module):
    def __init__(self, file_shape: tuple):
        super(UNet, self).__init__()

        # DownSampling
        self.down1 = UNetDown(file_shape[0], 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512)
        self.down5 = UNetDown(512, 512)
        self.down6 = UNetDown(512, 512)
        self.down7 = UNetDown(512, 512)
        self.down8 = UNetDown(512, 512, normalize=False)

        # UpSampling
        self.up1 = UNetUp(512, 512)
        self.up2 = UNetUp(1024, 512)
        self.up3 = UNetUp(1024, 512)
        self.up4 = UNetUp(1024, 512)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.output = nn.Sequential(
            ConvTranspose2d(128, 4, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.output(u7)
