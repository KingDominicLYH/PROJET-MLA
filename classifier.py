import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torch import optim

from src.models import Classifier
from src.tools import CelebADataset, Config

# 加载YAML配置
with open("parameter/parameters_classifier.yaml", "r") as f:
    params_dict = yaml.safe_load(f)

# 将YAML配置字典转换为Config对象
params = Config(params_dict)

# 参数配置
# n_attributes = 40
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据加载
train_dataset = CelebADataset(params.processed_file, split="train", transform=None, params=params)
valid_dataset = CelebADataset(params.processed_file, split="val", transform=None, params=params)

train_loader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=params.batch_size, shuffle=False)

# 模型、损失函数和优化器
model = Classifier().to(device)
criterion = nn.BCELoss()  # 适用于多标签分类问题
optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

# 训练过程
def train(model, train_loader, valid_loader, criterion, optimizer, n_epochs, device):
    # 用于保存最好的模型
    best_valid_loss = float('inf')

    for epoch in range(n_epochs):
        model.train()  # 设置为训练模式
        train_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        # 训练
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()  # 梯度清零
            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            train_loss += loss.item() * inputs.size(0)  # 累加损失
            preds = outputs > 0.5  # 二分类的阈值0.5
            correct_predictions += (preds == labels).sum().item()  # 计算正确预测
            total_predictions += labels.size(0) * labels.size(1)  # 总样本数

        train_loss /= len(train_loader.dataset)
        accuracy = correct_predictions / total_predictions

        # 验证
        model.eval()  # 设置为评估模式
        valid_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():  # 不计算梯度
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

                loss = criterion(outputs, labels)

                valid_loss += loss.item() * inputs.size(0)

                preds = outputs > 0.5
                correct_predictions += (preds == labels).sum().item()
                total_predictions += labels.size(0) * labels.size(1)

        valid_loss /= len(valid_loader.dataset)
        valid_accuracy = correct_predictions / total_predictions

        # 打印每个epoch的结果
        print(f'Epoch {epoch + 1}/{n_epochs}')
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {accuracy:.4f}')
        print(f'Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.4f}')

        # 保存最好的模型
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print('Model saved!')

# 运行训练
train(model, train_loader, valid_loader, criterion, optimizer, params.total_epochs, device)


