# Code taken from https://github.com/milesial/Pytorch-UNet
# Code adapted for 3D

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet3D_Base(nn.Module):
    def __init__(self, in_dim, out_dim, BN):
        super(UNet3D_Base, self).__init__()
        self.inc = inconv3d(in_dim, out_dim // 2, BN)
        self.down1 = down3d(out_dim // 2, out_dim // 4, BN)
        self.down2 = down3d(out_dim // 4, out_dim // 4, BN)
        # self.down3 = down3d(out_dim//8, out_dim//8, BN)

        # self.up1 = up3d(out_dim//4, out_dim//4, BN)
        self.up2 = up3d(out_dim // 2, out_dim // 2, BN)
        self.up3 = up3d(out_dim, out_dim, BN)
        self.outc = outconv3d(out_dim, out_dim)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        # x4 = self.down3(x3)
        # x = self.up1(x4, x3)
        x = self.up2(x3, x2)
        x = self.up3(x, x1)
        x = self.outc(x)
        return x3, x


###############################################################################
############################# UNET UTILS    ###################################
###############################################################################
class double_conv3d(nn.Module):
    """(conv => BN => LeakyReLU) * 2"""

    def __init__(self, in_ch, out_ch, BN):
        super(double_conv3d, self).__init__()
        in_ch = max(1, in_ch)
        out_ch = max(1, out_ch)
        kernel_size = 3
        if BN:
            self.conv = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm3d(out_ch),
                nn.LeakyReLU(inplace=True),
                nn.Conv3d(
                    out_ch, out_ch, kernel_size, padding=kernel_size // 2
                ),
                nn.BatchNorm3d(out_ch),
                nn.LeakyReLU(inplace=True),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
                nn.LeakyReLU(inplace=True),
                nn.Conv3d(
                    out_ch, out_ch, kernel_size, padding=kernel_size // 2
                ),
                nn.LeakyReLU(inplace=True),
            )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv3d(nn.Module):
    def __init__(self, in_ch, out_ch, BN):
        super(inconv3d, self).__init__()
        in_ch = max(1, in_ch)
        out_ch = max(1, out_ch)
        self.conv = double_conv3d(in_ch, out_ch, BN)

    def forward(self, x):
        x = self.conv(x)
        return x


class down3d(nn.Module):
    def __init__(self, in_ch, out_ch, BN):
        super(down3d, self).__init__()
        in_ch = max(1, in_ch)
        out_ch = max(1, out_ch)
        self.mpconv = nn.Sequential(
            nn.MaxPool3d(2), double_conv3d(in_ch, out_ch, BN)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up3d(nn.Module):
    def __init__(self, in_ch, out_ch, BN):
        super(up3d, self).__init__()
        in_ch = max(2, in_ch)  # 2 since up3d will get 2 outputs each time
        out_ch = max(1, out_ch)
        self.up3d = nn.Upsample(
            scale_factor=2, mode="trilinear", align_corners=True
        )
        self.conv = double_conv3d(in_ch, out_ch, BN)

    def forward(self, x1, x2):
        x1 = self.up3d(x1)

        # input is CHWD
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        diffZ = x2.size()[4] - x1.size()[4]

        x1 = F.pad(
            x1,
            (
                diffX // 2,
                diffX - diffX // 2,
                diffY // 2,
                diffY - diffY // 2,
                diffZ // 2,
                diffY - diffZ // 2,
            ),
        )

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv3d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv3d, self).__init__()
        in_ch = max(1, in_ch)
        out_ch = max(1, out_ch)
        self.conv = nn.Conv3d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
