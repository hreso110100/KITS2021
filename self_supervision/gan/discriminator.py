from torch import nn, cat
from torch.nn import Conv2d, Sequential, InstanceNorm2d
from torchsummary import summary

"""
PatchGan implementation
"""


class Discriminator(nn.Module):
    def __init__(self, file_shape: tuple):
        super(Discriminator, self).__init__()

        self.model = Sequential(
            *self.build_block(file_shape[0] * 2, 64, normalization=False),
            *self.build_block(64, 128),
            *self.build_block(128, 256),
            *self.build_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            Conv2d(512, 1, kernel_size=4, padding=1, stride=1, bias=False),
        )
        # summary(self.model, (file_shape[0] * 2, file_shape[1], file_shape[2]))

    def build_block(self, in_filters: int, out_filters: int, normalization=True):
        layers = [Conv2d(in_filters, out_filters, kernel_size=4, stride=2, padding=1)]

        if normalization:
            layers.append(InstanceNorm2d(num_features=out_filters, momentum=0.8))

        layers.append(nn.LeakyReLU(0.2, inplace=True))

        return layers

    def forward(self, input_a, input_b):
        img_input = cat((input_a, input_b), 1)

        return self.model(img_input).double()
