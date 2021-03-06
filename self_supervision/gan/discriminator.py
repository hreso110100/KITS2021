from torch import nn, cat
from torch.nn import Sequential, Conv3d, InstanceNorm3d

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
            nn.ReplicationPad3d((0, 1, 0, 1, 1, 0)),
            Conv3d(512, 1, kernel_size=4, padding=1, stride=1),
        )

    def build_block(self, in_filters: int, out_filters: int, normalization=True):
        layers = [Conv3d(in_filters, out_filters, kernel_size=4, stride=2, padding=1)]

        if normalization:
            layers.append(InstanceNorm3d(num_features=out_filters, momentum=0.8))

        layers.append(nn.LeakyReLU(0.2, inplace=True))

        return layers

    def forward(self, input_a, input_b):
        img_input = cat((input_a, input_b), 1)

        return self.model(img_input).double()
