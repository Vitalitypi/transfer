import torch
import torch.nn as nn

# 定义U-Net模型
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # 编码器部分
        self.enc1 = self.conv_block(3, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)

        # 中间部分
        self.middle = self.conv_block(256, 512)

        # 解码器部分
        self.dec3 = self.conv_block(512 + 256, 256)
        self.dec2 = self.conv_block(256 + 128, 128)
        self.dec1 = self.conv_block(128 + 64, 64)

        # 最后输出层
        self.final = nn.Conv2d(64, 3, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 编码器部分
        enc1 = self.enc1(x)
        enc2 = self.enc2(nn.MaxPool2d(2)(enc1))
        enc3 = self.enc3(nn.MaxPool2d(2)(enc2))

        # 中间部分
        middle = self.middle(nn.MaxPool2d(2)(enc3))

        # 解码器部分
        dec3 = self.dec3(torch.cat([nn.functional.interpolate(middle, scale_factor=2), enc3], dim=1))
        dec2 = self.dec2(torch.cat([nn.functional.interpolate(dec3, scale_factor=2), enc2], dim=1))
        dec1 = self.dec1(torch.cat([nn.functional.interpolate(dec2, scale_factor=2), enc1], dim=1))

        return self.final(dec1)


