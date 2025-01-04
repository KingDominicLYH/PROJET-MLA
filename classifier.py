import torch
import torch.nn as nn
import yaml
import os
from torch.utils.data import DataLoader
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm

from src.models import Classifier
from src.tools import CelebADataset, Config, get_optimizer

# 加载YAML配置
with open("parameter/parameters_classifier.yaml", "r") as f:
    params_dict = yaml.safe_load(f)
params = Config(params_dict)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据加载
train_dataset = CelebADataset(data_dir=params.processed_file, params=params, split="train")
valid_dataset = CelebADataset(data_dir=params.processed_file, params=params, split="val")

train_loader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=params.batch_size, shuffle=False, pin_memory=True)

# 模型、损失函数和优化器
model = Classifier(params).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = get_optimizer(model, params.optimizer)

# 获取当前时间戳并格式化为文件夹名称
current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
log_dir = f'Tensorboard/{current_time}'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
    print(f"Directory {log_dir} created.")
writer = SummaryWriter(log_dir=log_dir)


def train(model, train_loader, valid_loader, criterion, optimizer, n_epochs, device):
    """
    训练函数：前100个epoch不保存模型；
            从第101个epoch开始使用 Early-Stopping（基于验证准确率）。
    """

    # ========= 新增：Early-Stopping 所需变量 =========
    best_valid_accuracy = 0.0       # 记录历史最佳验证准确率
    patience = 50                   # 当验证准确率连续 50 个 epoch 没有提升就提前停止
    no_improvement_count = 0        # 记录连续多少个 epoch 未提升
    save_start_epoch = 150          # 指定从第几个 epoch 开始进行模型保存和 Early-Stopping

    # 计算每个 epoch 的迭代次数（与原逻辑相同）
    num_iterations = params.total_train_samples // params.batch_size
    num_valid_iterations = params.total_valid_samples // params.batch_size

    for epoch in range(n_epochs):
        print(f'Starting Epoch {epoch + 1}/{n_epochs}')

        # ======= 训练阶段 =======
        model.train()
        train_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        num_samples = 0

        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{n_epochs}",
                                 dynamic_ncols=True, total=num_iterations)

        for i, (inputs, labels) in enumerate(train_loader_tqdm):
            if i >= num_iterations:
                break  # 超过指定的训练样本数量后跳出
            label_indices = labels.argmax(dim=-1)
            inputs, label_indices = inputs.to(device), label_indices.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.permute(0, 2, 1), label_indices)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=-1)
            correct_predictions += (preds == label_indices).sum().item()
            total_predictions += label_indices.numel()
            num_samples += params.batch_size

            train_loader_tqdm.set_postfix({'Loss': f'{loss.item():.4f}'})

        train_loss /= num_samples
        train_accuracy = correct_predictions / total_predictions

        # ======= 验证阶段 =======
        model.eval()
        valid_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        num_samples = 0

        valid_loader_tqdm = tqdm(valid_loader, desc=f"Validating Epoch {epoch + 1}/{n_epochs}",
                                 dynamic_ncols=True, total=num_valid_iterations)
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(valid_loader_tqdm):
                if i >= num_valid_iterations:
                    break
                label_indices = labels.argmax(dim=-1)
                inputs, label_indices = inputs.to(device), label_indices.to(device)
                outputs = model(inputs)

                loss = criterion(outputs.permute(0, 2, 1), label_indices)
                valid_loss += loss.item() * inputs.size(0)

                preds = outputs.argmax(dim=-1)
                correct_predictions += (preds == label_indices).sum().item()
                total_predictions += label_indices.numel()
                num_samples += params.batch_size

        valid_loss /= num_samples
        valid_accuracy = correct_predictions / total_predictions

        # ====== 打印本轮训练与验证信息 ======
        print(f'Epoch {epoch + 1}/{n_epochs}')
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
        print(f'Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.4f}')

        # 写入 TensorBoard
        writer.add_scalar('Train Loss', train_loss, epoch)
        writer.add_scalar('Train Accuracy', train_accuracy, epoch)
        writer.add_scalar('Valid Loss', valid_loss, epoch)
        writer.add_scalar('Valid Accuracy', valid_accuracy, epoch)

        # ====== 如果保存目录不存在则创建 ======
        if not os.path.exists(params.save_dir):
            os.makedirs(params.save_dir)
            print(f"Directory {params.save_dir} created.")

        # =========== 关键逻辑：前100个epoch不保存模型，从第101个epoch开始Early-Stopping ===========
        if epoch + 1 <= save_start_epoch:
            # 前 100 个 epoch 只训练，不保存模型，也不进行 early-stopping 统计
            continue
        else:
            # 只有当验证准确率比之前的最优值更高时，才保存模型并归零 no_improvement_count
            if valid_accuracy > best_valid_accuracy:
                best_valid_accuracy = valid_accuracy
                no_improvement_count = 0
                save_path = os.path.join(params.save_dir, "best_model_epoch_{epoch+1}.pth")
                torch.save(model.state_dict(), save_path)
                print(f"[Epoch {epoch + 1}] Model saved to {save_path} with valid acc {valid_accuracy:.4f}!")
            else:
                # 如果验证准确率没有提升，则计数 +1
                no_improvement_count += 1
                print(f"[Epoch {epoch + 1}] No improvement. Count {no_improvement_count}/{patience}.")

                # 如果连续 50 个 epoch 无提升，则停止训练
                if no_improvement_count >= patience:
                    print("Early stopping triggered!")
                    break


# 运行训练
train(model, train_loader, valid_loader, criterion, optimizer, params.total_epochs, device)
