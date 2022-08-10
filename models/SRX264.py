import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.gdn import GDN
import math

class SRX264(nn.Module):
    def __init__(self, scale_factor = 2, maps = 96, in_nc=13, out_nc=9) -> None:
        super(SRX264, self).__init__()
        upsample_block_num = int(math.log(scale_factor, 2))
        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(in_nc, maps, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(maps)
        self.block3 = ResidualBlock(maps)
        self.block4 = ResidualBlock(maps)
        self.block5 = ResidualBlock(maps)
        self.block6 = ResidualBlock(maps)
        self.block7 = nn.Sequential(
            nn.Conv2d(maps, maps, kernel_size=3, padding=1),
            GDN(maps)
        )
        block8 = [UpsampleBLock(maps, 2) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(maps, out_nc, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)
        
    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6+block2)
        block8 = self.block8(block7)
        return (torch.tanh(block8) + 1) / 2 

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.gdn1 = GDN(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.gdn2 = GDN(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.gdn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.gdn2(residual)

        return x + residual
    
class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x
