import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, n_attributes):
        super(Discriminator, self).__init__()
        self.n_attributes = n_attributes

        # C512 卷积层
        self.conv = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),  # 降采样
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3)  # 在卷积层后添加 Dropout，丢弃率为 0.3
        )

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(512 * 2 * 2, 512),  # 输入尺寸为 (512, 2, 2)，展平后为 512*2*2
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),# 在第一层全连接层后添加 Dropout
            nn.Linear(512, n_attributes),  # 输出尺寸为 n_attributes
            nn.Sigmoid()  # 每个属性值归一化到 [0, 1]，表示分类概率
        )

    def forward(self, z):
        """
        参数：
        z: 输入特征图 (batch_size, 512, 2, 2)，来自编码器
        """
        # 通过卷积层提取特征
        x = self.conv(z)
        # 展平后输入全连接层
        x = x.view(x.size(0), -1)  # 将 (batch_size, 512, 2, 2) 转为 (batch_size, 512*2*2)
        # 通过全连接层输出预测
        output = self.fc(x)
        return output

# 测试判别器
if __name__ == "__main__":
    discriminator = Discriminator(n_attributes=5)  # 假设有 5 个属性
    z = torch.randn(1, 512, 2, 2)  # 模拟编码器的输出
    output = discriminator(z)
    print("Output shape:", output.shape)  # 输出形状应为 (1, 5)，每个属性一个概率值
