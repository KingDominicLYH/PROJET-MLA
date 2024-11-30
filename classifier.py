import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime  # 导入datetime模块
from tqdm import tqdm  # 导入进度条库

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

# 数据加载部分
train_dataset = CelebADataset(data_dir=params.processed_file, params=params, split="train")
valid_dataset = CelebADataset(data_dir=params.processed_file, params=params, split="val")

train_loader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=params.batch_size, shuffle=False, pin_memory=True)

# 模型、损失函数和优化器
model = Classifier(params).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

# 获取当前时间戳并格式化为文件夹名称
current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
log_dir = f'Tensorboard/{current_time}' # 动态生成TensorBoard日志目录
writer = SummaryWriter(log_dir=log_dir) # 初始化TensorBoard的SummaryWriter

# 训练过程
def train(model, train_loader, valid_loader, criterion, optimizer, n_epochs, device):
    # 用于保存最好的模型
    best_valid_loss = float('inf')

    # 计算每个 epoch 中的迭代次数
    num_iterations = params.total_train_samples // params.batch_size
    num_valid_iterations = params.total_valid_samples // params.batch_size

    for epoch in range(n_epochs):
        print(f'Starting Epoch {epoch + 1}/{n_epochs}')
        model.train()  # 设置为训练模式
        train_loss = 0
        correct_predictions = 0
        total_predictions = 0

        # 创建进度条
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{n_epochs}", dynamic_ncols=True, total=num_iterations)

        # 训练一个 epoch 时，只迭代 num_iterations 次
        for i, (inputs, labels) in enumerate(train_loader_tqdm):
            if i >= num_iterations:
                break  # 超过50000个样本后，终止训练

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()  # 梯度清零

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)  # 累加损失
            preds = outputs > 0.5  # 二分类的阈值0.5

            # 计算准确度
            correct_predictions += (preds == labels).sum().item()
            total_predictions += labels.numel()

            # 更新进度条
            train_loader_tqdm.set_postfix({'Loss': f'{loss.item():.4f}'})

        train_loss /= len(train_loader.dataset)
        accuracy = correct_predictions / total_predictions

        # 验证
        model.eval()  # 设置为评估模式
        valid_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        # 创建验证进度条
        valid_loader_tqdm = tqdm(valid_loader, desc=f"Validating Epoch {epoch + 1}/{n_epochs}", dynamic_ncols=True)

        with torch.no_grad():  # 不计算梯度
            for inputs, labels in valid_loader_tqdm:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

                loss = criterion(outputs, labels)

                valid_loss += loss.item() * inputs.size(0)

                preds = outputs > 0.5

                # 计算准确度
                correct_predictions += (preds == labels).sum().item()
                total_predictions += labels.numel()

        valid_loss /= len(valid_loader.dataset)
        valid_accuracy = correct_predictions / total_predictions

        # 打印每个epoch的结果
        print(f'Epoch {epoch + 1}/{n_epochs}')
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {accuracy:.4f}')
        print(f'Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.4f}')

        writer.add_scalar('Train Loss', train_loss, epoch)
        writer.add_scalar('Train Accuracy', accuracy, epoch)
        writer.add_scalar('Valid Loss', valid_loss, epoch)
        writer.add_scalar('Valid Accuracy', valid_accuracy, epoch)

        # 保存最好的模型
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print('Model saved!')

# 运行训练
train(model, train_loader, valid_loader, criterion, optimizer, params.total_epochs, device)


