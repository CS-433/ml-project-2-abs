import torch
import torch.nn as nn
from collections import OrderedDict


# Unet model derived from https://github.com/mateuszbuda/brain-segmentation-pytorch

class UNet(nn.Module):

    def __init__(self, n_channels=3, n_classes=1, init_features=32):
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = UNet._block(n_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=n_classes, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )

class WNet(nn.Module):

    def __init__(self, n_channels=3, n_classes=1, init_features=32):
        super(WNet, self).__init__()

        features = init_features
        self.encoder1 = WNet._block(n_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = WNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = WNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = WNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck1 = WNet._block(features * 8, features * 16, name="bottleneck")
        self.bottleneck2 = WNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = WNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = WNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = WNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = WNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=n_classes, kernel_size=1
        )

        self.encoder1_ = WNet._block(n_classes, features, name="enc1_")

    def forward(self, x):

        # First V
        enc11 = self.encoder1(x)
        enc12 = self.encoder2(self.pool1(enc11))
        enc13 = self.encoder3(self.pool2(enc12))
        enc14 = self.encoder4(self.pool3(enc13))
        bottleneck = self.bottleneck1(self.pool4(enc14))
        dec14 = self.upconv4(bottleneck)
        dec14 = torch.cat((dec14, enc14), dim=1)
        dec14 = self.decoder4(dec14)
        dec13 = self.upconv3(dec14)
        dec13 = torch.cat((dec13, enc13), dim=1)
        dec13 = self.decoder3(dec13)
        dec12 = self.upconv2(dec13)
        dec12 = torch.cat((dec12, enc12), dim=1)
        dec12 = self.decoder2(dec12)
        dec11 = self.upconv1(dec12)
        dec11 = torch.cat((dec11, enc11), dim=1)
        dec11 = self.decoder1(dec11)

        # Second V
        enc21 = self.encoder1_(torch.relu(self.conv(dec11))) # TODO: try w/o self.conv (more features) # TODO: try sigmoid
        enc22 = self.encoder2(self.pool1(enc21))
        enc23 = self.encoder3(self.pool2(enc22))
        enc24 = self.encoder4(self.pool3(enc23))
        bottleneck = self.bottleneck2(self.pool4(enc24))
        dec24 = self.upconv4(bottleneck)
        dec24 = torch.cat((dec24, enc24), dim=1) # TODO: try enc14, enc13, etc.
        dec24 = self.decoder4(dec24)
        dec23 = self.upconv3(dec24)
        dec23 = torch.cat((dec23, enc23), dim=1)
        dec23 = self.decoder3(dec23)
        dec22 = self.upconv2(dec23)
        dec22 = torch.cat((dec22, enc22), dim=1)
        dec22 = self.decoder2(dec22)
        dec21 = self.upconv1(dec22)
        dec21 = torch.cat((dec21, enc21), dim=1)
        dec21 = self.decoder1(dec21)

        return torch.sigmoid(self.conv(dec21))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )