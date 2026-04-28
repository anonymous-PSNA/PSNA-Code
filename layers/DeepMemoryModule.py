import torch
import torch.nn as nn
import torch.nn.functional as F

class MagPhaseFusionGating(nn.Module):
    def __init__(self, channels):
        super(MagPhaseFusionGating, self).__init__()
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.InstanceNorm2d(channels, affine=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        mask = self.fusion_conv(x)
        return x * mask + x

class PhaseAwareUNet2D(nn.Module):
    def __init__(self, base_dim=32):
        super(PhaseAwareUNet2D, self).__init__()
        self.base_dim = base_dim
        in_c = 2

        self.enc1 = nn.Sequential(
            nn.Conv2d(in_c, base_dim, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_dim, affine=True),
            nn.GELU(),
            nn.Dropout2d(0.2)
        )
        self.gate1 = MagPhaseFusionGating(base_dim)

        self.enc2 = nn.Sequential(
            nn.Conv2d(base_dim, base_dim*2, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_dim*2, affine=True),
            nn.GELU(),
            nn.Dropout2d(0.2)
        )
        self.gate2 = MagPhaseFusionGating(base_dim*2)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_dim*2, base_dim*2, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_dim*2, affine=True),
            nn.GELU(),
            nn.Dropout2d(0.4)
        )

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.dec1 = nn.Sequential(
            nn.Conv2d(base_dim*4, base_dim, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_dim, affine=True),
            nn.GELU(),
            nn.Dropout2d(0.2)
        )

        self.dec2 = nn.Sequential(
            nn.Conv2d(base_dim*2, base_dim, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_dim, affine=True),
            nn.GELU(),
            nn.Dropout2d(0.2),
            nn.Conv2d(base_dim, 2, kernel_size=1)
        )

    def forward(self, x):

        e1 = self.gate1(self.enc1(x))
        p1 = self.pool(e1)

        e2 = self.gate2(self.enc2(p1))
        p2 = self.pool(e2)

        b = self.bottleneck(p2)

        u1 = self.upsample(b)
        if u1.shape[-2:] != e2.shape[-2:]:
            u1 = F.interpolate(u1, size=e2.shape[-2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([u1, e2], dim=1))

        u2 = self.upsample(d1)
        if u2.shape[-2:] != e1.shape[-2:]:
            u2 = F.interpolate(u2, size=e1.shape[-2:], mode='bilinear', align_corners=False)
        out = self.dec2(torch.cat([u2, e1], dim=1))
        a = out + x
        return a
