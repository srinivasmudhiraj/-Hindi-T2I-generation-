import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict



class NetG(nn.Module):
    def __init__(self, ngf=64, nz=100):
        super(NetG, self).__init__()
        self.ngf = ngf

        # Fully connected projection
        self.fc = nn.Linear(nz, ngf * 8 * 4 * 4)

        # Generator blocks (progressive upsampling)
        self.block0 = G_Block(ngf * 8, ngf * 8)  # 4x4
        self.block1 = G_Block(ngf * 8, ngf * 8)  # 8x8
        self.block2 = G_Block(ngf * 8, ngf * 8)  # 16x16
        self.block3 = G_Block(ngf * 8, ngf * 8)  # 32x32
        self.block4 = G_Block(ngf * 8, ngf * 4)  # 64x64
        self.block5 = G_Block(ngf * 4, ngf * 2)  # 128x128
        self.block6 = G_Block(ngf * 2, ngf)      # 256x256

        # Final image convolution
        self.conv_img = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z, cond):
        x_out = self.fc(z).view(z.size(0), 8 * self.ngf, 4, 4)

        for block in [self.block0, self.block1, self.block2,
                      self.block3, self.block4, self.block5, self.block6]:
            x_out = block(x_out, cond) if block is self.block0 else F.interpolate(x_out, scale_factor=2, mode="nearest").contiguous()
            x_out = block(x_out, cond)

        return self.conv_img(x_out)


class G_Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(G_Block, self).__init__()
        self.need_sc = in_ch != out_ch
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.c2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)

        self.affine0 = affine(in_ch)
        self.affine1 = affine(in_ch)
        self.affine2 = affine(out_ch)
        self.affine3 = affine(out_ch)

        self.gamma = nn.Parameter(torch.zeros(1))
        if self.need_sc:
            self.c_sc = nn.Conv2d(in_ch, out_ch, 1, stride=1)

    def forward(self, x, y=None):
        return self.shortcut(x) + self.gamma * self.residual(x, y)

    def shortcut(self, x):
        return self.c_sc(x) if self.need_sc else x

    def residual(self, x, y=None):
        feat = F.leaky_relu(self.affine0(x, y), 0.2, inplace=True)
        feat = F.leaky_relu(self.affine1(feat, y), 0.2, inplace=True)
        feat = self.c1(feat)

        feat = F.leaky_relu(self.affine2(feat, y), 0.2, inplace=True)
        feat = F.leaky_relu(self.affine3(feat, y), 0.2, inplace=True)
        return self.c2(feat)


class affine(nn.Module):
    def __init__(self, num_features):
        super(affine, self).__init__()

        self.fc_gamma = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Linear(256, num_features)
        )
        self.fc_beta = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Linear(256, num_features)
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.fc_gamma[2].weight)
        nn.init.ones_(self.fc_gamma[2].bias)
        nn.init.zeros_(self.fc_beta[2].weight)
        nn.init.zeros_(self.fc_beta[2].bias)

    def forward(self, x, y=None):
        gamma = self.fc_gamma(y).unsqueeze(-1).unsqueeze(-1).expand_as(x)
        beta = self.fc_beta(y).unsqueeze(-1).unsqueeze(-1).expand_as(x)
        return gamma * x + beta


class D_GET_LOGITS(nn.Module):
    def __init__(self, ndf):
        super(D_GET_LOGITS, self).__init__()
        self.joint_conv = nn.Sequential(
            nn.Conv2d(ndf * 16 + 256, ndf * 2, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, 1, 4, 1, 0, bias=False),
        )

    def forward(self, x, y):
        cond = y.view(-1, 256, 1, 1).repeat(1, 1, 4, 4)
        h = torch.cat((x, cond), dim=1)
        return self.joint_conv(h)


class NetD(nn.Module):
    def __init__(self, ndf):
        super(NetD, self).__init__()
        self.conv_img = nn.Conv2d(3, ndf, 3, 1, 1)  # 256x256

        self.block0 = resD(ndf, ndf * 2)    # 128x128
        self.block1 = resD(ndf * 2, ndf * 4)  # 64x64
        self.block2 = resD(ndf * 4, ndf * 8)  # 32x32
        self.block3 = resD(ndf * 8, ndf * 16) # 16x16
        self.block4 = resD(ndf * 16, ndf * 16) # 8x8
        self.block5 = resD(ndf * 16, ndf * 16) # 4x4

        self.COND_DNET = D_GET_LOGITS(ndf)

    def forward(self, x):
        x_out = self.conv_img(x)
        for block in [self.block0, self.block1, self.block2, self.block3, self.block4, self.block5]:
            x_out = block(x_out)
        return x_out


class resD(nn.Module):
    def __init__(self, fin, fout, downsample=True):
        super(resD, self).__init__()
        self.downsample = downsample
        self.need_sc = fin != fout

        self.conv_r = nn.Sequential(
            nn.Conv2d(fin, fout, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(fout, fout, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        if self.need_sc:
            self.conv_s = nn.Conv2d(fin, fout, 1, 1, 0)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, c=None):
        return self.shortcut(x) + self.gamma * self.residual(x)

    def shortcut(self, x):
        if self.need_sc:
            x = self.conv_s(x)
        return F.avg_pool2d(x, 2) if self.downsample else x

    def residual(self, x):
        return self.conv_r(x)
