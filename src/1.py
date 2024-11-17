import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # 定义卷积层，使用 Conv2d -> BatchNorm2d -> LeakyReLU 的组合
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1),  # 输出大小减半
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        # 输入 x 通过 encoder 网络，生成潜在表示 z
        z = self.encoder(x)
        return z


# 测试编码器
if __name__ == "__main__":
    encoder = Encoder()
    # 假设输入图片大小为 256x256
    x = torch.randn(1, 3, 256, 256)  # Batch size 1, 3 通道 (RGB), 256x256 图像
    z = encoder(x)
    print("Output shape:", z.shape)  # 输出形状应该是 (1, 512, 2, 2)

    print(encoder)

