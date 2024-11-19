import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, n_attributes):
        super(Decoder, self).__init__()
        self.n_attributes = n_attributes
        # 计算附加属性通道数
        attribute_channels = 2 * n_attributes  # 每个属性用 [1, 0] 或 [0, 1] 表示

        # 定义反卷积层，使用 TransposeConv2d -> BatchNorm2d -> ReLU 的组合
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512 + attribute_channels, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(512 + attribute_channels, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256 + attribute_channels, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128 + attribute_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64 + attribute_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32 + attribute_channels, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(16 + attribute_channels, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # 输出归一化到 [-1, 1]
        )

    def forward(self, z, attributes):
        """
        参数：
        z: 编码器的潜在表示 (batch_size, 512, 2, 2)
        attributes: 属性向量 (batch_size, n_attributes)，one-hot 编码
        """
        # 将属性向量扩展为与 z 相同的空间尺寸
        batch_size, _, h, w = z.shape
        attributes = attributes.view(batch_size, -1, 1, 1)  # 调整为 (batch_size, 2n, 1, 1)
        attributes = attributes.expand(batch_size, -1, h, w)  # 扩展为 (batch_size, 2n, h, w)

        # 将属性向量附加到每一层输入
        x = torch.cat([z, attributes], dim=1)  # (batch_size, 512 + 2n, h, w)
        for layer in self.decoder:
            if isinstance(layer, nn.ConvTranspose2d):
                x = torch.cat([x, attributes], dim=1)  # 每层都附加属性向量
            x = layer(x)
        return x

# 测试解码器
if __name__ == "__main__":
    decoder = Decoder(n_attributes=5)  # 假设有 5 个属性
    z = torch.randn(1, 512, 2, 2)  # 编码器输出的潜在表示
    attributes = torch.randint(0, 2, (1, 5))  # 随机生成属性向量
    output = decoder(z, attributes)
    print("Output shape:", output.shape)  # 输出形状应为 (1, 3, 256, 256)

