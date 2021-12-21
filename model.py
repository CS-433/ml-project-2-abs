import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision.models.resnet import BasicBlock, Bottleneck


# Unet model inspired by https://github.com/mateuszbuda/brain-segmentation-pytorch
class UNet(nn.Module):

    def __init__(self, n_channels=3, n_classes=1, init_features=32):
        super(UNet, self).__init__()

        block = UNet._block
        features = init_features
        
        # Encoder side
        self.encoder1 = block(n_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = block(features * 8, features * 16, name="bottleneck")

        # Decoder side
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = block(features * 2, features, name="dec1")

        # Output
        self.conv = nn.Conv2d(in_channels=features, out_channels=n_classes, kernel_size=1)

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
    def _block(in_channels, features, name, **kwargs):
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


class UNet06(nn.Module):

    def __init__(self, n_channels=3, n_classes=1, init_features=32):
        super(UNet06, self).__init__()

        block = UNet06._block

        # Avoiding more than 512 features
        features1 = min(init_features, 512)
        features2 = min(features1 * 2, 512)
        features3 = min(features2 * 2, 512)
        features4 = min(features3 * 2, 512)
        features5 = min(features4 * 2, 512)
        features6 = min(features5 * 2, 512)

        # Encoder side
        self.encoder1 = block(n_channels, features1, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)        
        self.encoder2 = block(features1, features2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = block(features2, features3, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = block(features3, features4, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder5 = block(features4, features5, name="enc5")
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder6 = block(features5, features6, name="enc6")

        # Bottleneck
        self.bottleneck = block(features6, features6, name="bottleneck")
        
        # Decoder side
        self.decoder6 = block(features6 * 2, features6, name="dec6")  
        self.upconv5 = nn.ConvTranspose2d(features6, features5, kernel_size=2, stride=2)
        self.decoder5 = block(features5 * 2, features5, name="dec5")  
        self.upconv4 = nn.ConvTranspose2d(features5, features4, kernel_size=2, stride=2)
        self.decoder4 = block(features4 * 2, features4, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(features4, features3, kernel_size=2, stride=2)
        self.decoder3 = block(features3 * 2, features3, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(features3, features2, kernel_size=2, stride=2)
        self.decoder2 = block(features2 * 2, features2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(features2, features1, kernel_size=2, stride=2)
        self.decoder1 = block(features1 * 2, features1, name="dec1")

        # Output
        self.conv = nn.Conv2d(in_channels=features1, out_channels=n_classes, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        enc5 = self.encoder5(self.pool4(enc4))
        enc6 = self.encoder6(self.pool5(enc5))
        bottleneck = self.bottleneck(enc6)
        dec6 = bottleneck # no need  to upconv because pool6 keeps the size of the input
        dec6 = torch.cat((dec6, enc6), dim=1)
        dec6 = self.decoder6(dec6)
        dec5 = self.upconv5(dec6)
        dec5 = torch.cat((dec5, enc5), dim=1)
        dec5 = self.decoder5(dec5)
        dec4 = self.upconv4(dec5)
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
    def _block(in_channels, features, name, **kwargs):
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


class WNet0404(nn.Module):

    def __init__(self, n_channels=3, n_classes=1, init_features=32):
        super(WNet0404, self).__init__()

        features = init_features

        # First U
        self.encoder11 = WNet0404._block(n_channels, features, name="enc11")
        self.pool11 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder12 = WNet0404._block(features, features * 2, name="enc12")
        self.pool12 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder13 = WNet0404._block(features * 2, features * 4, name="enc13")
        self.pool13 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder14 = WNet0404._block(features * 4, features * 8, name="enc14")
        self.pool14 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck1 = WNet0404._block(features * 8, features * 16, name="bottleneck1")
        self.upconv14 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder14 = WNet0404._block((features * 8) * 2, features * 8, name="dec14")
        self.upconv13 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder13 = WNet0404._block((features * 4) * 2, features * 4, name="dec13")
        self.upconv12 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder12 = WNet0404._block((features * 2) * 2, features * 2, name="dec12")
        self.upconv11 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder11 = WNet0404._block(features * 2, features, name="dec11")
        self.conv1 = nn.Conv2d(in_channels=features, out_channels=n_classes, kernel_size=1)

        # Second U
        self.encoder21 = WNet0404._block(n_classes, features, name="enc21")
        self.pool21 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder22 = WNet0404._block(features, features * 2, name="enc22")
        self.pool22 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder23 = WNet0404._block(features * 2, features * 4, name="enc23")
        self.pool23 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder24 = WNet0404._block(features * 4, features * 8, name="enc24")
        self.pool24 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck2 = WNet0404._block(features * 8, features * 16, name="bottleneck2")
        self.upconv24 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder24 = WNet0404._block((features * 8) * 2, features * 8, name="dec24")
        self.upconv23 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder23 = WNet0404._block((features * 4) * 2, features * 4, name="dec23")
        self.upconv22 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder22 = WNet0404._block((features * 2) * 2, features * 2, name="dec22")
        self.upconv21 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder21 = WNet0404._block(features * 2, features, name="dec21")
        self.conv2 = nn.Conv2d(in_channels=features, out_channels=n_classes, kernel_size=1)

    def forward(self, x):

        # First U
        enc11 = self.encoder11(x)
        enc12 = self.encoder12(self.pool11(enc11))
        enc13 = self.encoder13(self.pool12(enc12))
        enc14 = self.encoder14(self.pool13(enc13))
        bottleneck1 = self.bottleneck1(self.pool14(enc14))
        dec14 = self.upconv14(bottleneck1)
        dec14 = torch.cat((dec14, enc14), dim=1)
        dec14 = self.decoder14(dec14)
        dec13 = self.upconv13(dec14)
        dec13 = torch.cat((dec13, enc13), dim=1)
        dec13 = self.decoder13(dec13)
        dec12 = self.upconv12(dec13)
        dec12 = torch.cat((dec12, enc12), dim=1)
        dec12 = self.decoder12(dec12)
        dec11 = self.upconv11(dec12)
        dec11 = torch.cat((dec11, enc11), dim=1)
        dec11 = self.decoder11(dec11)

        # Second U
        enc21 = self.encoder21(torch.sigmoid(self.conv1(dec11)))
        enc22 = self.encoder22(self.pool21(enc21))
        enc23 = self.encoder23(self.pool22(enc22))
        enc24 = self.encoder24(self.pool23(enc23))
        bottleneck2 = self.bottleneck2(self.pool24(enc24))
        dec24 = self.upconv24(bottleneck2)
        dec24 = torch.cat((dec24, enc14), dim=1)
        dec24 = self.decoder24(dec24)
        dec23 = self.upconv23(dec24)
        dec23 = torch.cat((dec23, enc13), dim=1)
        dec23 = self.decoder23(dec23)
        dec22 = self.upconv22(dec23)
        dec22 = torch.cat((dec22, enc12), dim=1)
        dec22 = self.decoder22(dec22)
        dec21 = self.upconv21(dec22)
        dec21 = torch.cat((dec21, enc11), dim=1)
        dec21 = self.decoder21(dec21)

        return torch.sigmoid(self.conv2(dec21))

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
