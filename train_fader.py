import os
import torch
import yaml
from datetime import datetime

from torch import optim, nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.models import AutoEncoder, Discriminator, Classifier
from src.tools import CelebADataset, Config

# Load configuration parameters from the YAML file
with open("parameter/parameters.yaml", "r") as f:
    params_dict = yaml.safe_load(f)
# 将YAML配置字典转换为Config对象
params = Config(params_dict)

# Load configuration parameters from the YAML file
with open("parameter/parameters_classifier.yaml", "r") as f:
    params_dict_classifier = yaml.safe_load(f)
# 将YAML配置字典转换为Config对象
params_classifier = Config(params_dict_classifier)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data preparation: load training and validation datasets
train_dataset = CelebADataset(data_dir=params.processed_file, params=params, split="train")
valid_dataset = CelebADataset(data_dir=params.processed_file, params=params, split="val")

train_loader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=params.batch_size, shuffle=False, pin_memory=True)

print("Train dataset size:", len(train_dataset))
print("Validation dataset size:", len(valid_dataset))

target_attribute_indices = [
    params.ALL_ATTR.index(attr) for attr in params.target_attribute_list
]
print(f"Target attribute indices: {target_attribute_indices}")

# Initialize models
autoencoder = AutoEncoder(params).to(device)
discriminator = Discriminator(params).to(device)

# Load the pre-trained classifier model
classifier = Classifier(params_classifier).to(device)
incompatible_keys = classifier.load_state_dict(torch.load(params.model_path, map_location=device))
classifier.eval()

# Setup optimizers
autoencoder_optimizer = optim.Adam(
        autoencoder.parameters(),
        lr=params.learning_rate,
        betas=(0.5, 0.999)
    )

discriminator_optimizer = optim.Adam(
        discriminator.parameters(),
        lr=params.learning_rate,
        betas=(0.5, 0.999)
    )

# Define loss functions
reconstruction_criterion = nn.MSELoss()
classification_criterion = nn.CrossEntropyLoss()

# 获取当前时间戳并格式化为文件夹名称
current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
log_dir = f'Tensorboard/train_{current_time}'  # 动态生成 TensorBoard 日志目录

# 检查并创建目录
os.makedirs(log_dir, exist_ok=True)  # 如果目录不存在，则创建
print(f"TensorBoard logs saving to: {log_dir}")

# 初始化 TensorBoard 的 SummaryWriter
writer = SummaryWriter(log_dir=log_dir)

os.makedirs(params.model_output_path, exist_ok=True)

def train():
    best_swap_accuracy = 0.0

    # 计算每个 epoch 中的迭代次数
    num_iterations = params.total_train_samples // params.batch_size
    step_counter = 0  # 用于记录当前训练步数

    total_epochs = params.total_epochs
    for epoch in range(1, total_epochs + 1):
        print(f'Starting Epoch {epoch}/{total_epochs}')
        autoencoder.train()
        discriminator.train()

        # 初始化用于统计的变量
        total_recon_loss = 0.0
        total_discriminator_loss = 0.0
        total_adversarial_loss = 0.0
        num_samples = 0

        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs}", dynamic_ncols=True, total=num_iterations)
        for step, (images, labels) in enumerate(train_loader_tqdm):
            if step >= num_iterations:
                break  # 超过50000个样本后，终止训练
            label_indices = labels.argmax(dim=-1)  # [N, 40, 2] -> [N, 40]
            images, labels, label_indices = images.to(device), labels.to(device), label_indices.to(device)

            # 动态调整 discriminator_loss_weight
            max_steps = 500000
            params.latent_discriminator_loss_weight = min(
                0.001,  # 最大值
                (step_counter / max_steps) * 0.001  # 按比例增长
            )

            # (i) 训练判别器
            latent = autoencoder.encoder(images)
            pred_labels_real = discriminator(latent.detach())
            loss_discriminator = classification_criterion(pred_labels_real.permute(0, 2, 1), label_indices)

            discriminator_optimizer.zero_grad()
            loss_discriminator.backward()
            discriminator_optimizer.step()

            # (ii) 训练自编码器
            latent = autoencoder.encoder(images)
            reconstructed_images = autoencoder.decoder(latent, labels)
            recon_loss = reconstruction_criterion(reconstructed_images, images)

            # 翻转标签 => 1 - labels
            pred_labels_fake = discriminator(latent)
            adversarial_loss = classification_criterion(pred_labels_fake.permute(0, 2, 1), 1 - label_indices)

            ae_loss = params.autoencoder_loss_weight * recon_loss + params.latent_discriminator_loss_weight * adversarial_loss

            autoencoder_optimizer.zero_grad()
            ae_loss.backward()
            autoencoder_optimizer.step()

            # 累加每个 step 的损失
            total_recon_loss += recon_loss.item() * params.batch_size
            total_discriminator_loss += loss_discriminator.item() * params.batch_size
            total_adversarial_loss += adversarial_loss.item() * params.batch_size
            num_samples += params.batch_size

            # (iii) 日志 & 进度条
            train_loader_tqdm.set_postfix({
                "Recon": recon_loss.item(),
                "D_loss": loss_discriminator.item(),
                "Adv": adversarial_loss.item()
            })

            step_counter += params.batch_size

        # 计算 epoch 平均损失
        print(step)
        avg_recon_loss = total_recon_loss / num_samples
        avg_discriminator_loss = total_discriminator_loss / num_samples
        avg_adversarial_loss = total_adversarial_loss / num_samples

        # 记录到 TensorBoard
        writer.add_scalar("Epoch_Avg_Reconstruction_Loss", avg_recon_loss, epoch)
        writer.add_scalar("Epoch_Avg_Discriminator_Loss", avg_discriminator_loss, epoch)
        writer.add_scalar("Epoch_Avg_Adversarial_Loss", avg_adversarial_loss, epoch)

        # f) 验证
        autoencoder.eval()
        classifier.eval()  # 验证属性替换时需要分类器
        # total_val_loss = 0.0
        swap_accuracy = 0.0
        num_samples = 0


        with torch.no_grad():
            for images, labels in valid_loader:
                # 将数据移动到设备
                images, labels = images.to(device), labels.to(device)

                # 替换属性（例如：切换第 i 个属性，假设属性为“眼镜”）
                # 从 [32, 1, 2] 转换为 [32, 1]，得到单个类别索引（0 或 1）
                modified_labels = labels.clone()  # [32, 1, 2]
                modified_labels[:, :, 0], modified_labels[:, :, 1] = modified_labels[:, :, 1], modified_labels[:, :,0].clone()

                swapped_images = autoencoder(images, modified_labels)

                modified_labels = modified_labels.argmax(dim=2)  # [32, 1]

                # 使用分类器验证生成图像是否符合修改后的属性
                predicted_labels = classifier(swapped_images).argmax(dim=2)  # 分类器输出形状为 [batch_size, n_attributes, 2]
                target_predicted_labels = predicted_labels[:, target_attribute_indices]  # 只保留目标属性的预测值

                # 计算属性替换的准确性
                swap_accuracy += (target_predicted_labels == modified_labels).sum().item()

                num_samples += images.size(0)

        # 属性替换准确率
        swap_accuracy = swap_accuracy / num_samples

        # 使用 TensorBoard 记录验证指标
        writer.add_scalar("Validation/Attribute_Swap_Accuracy", swap_accuracy, epoch)

        # 打印验证结果
        print(f"[Epoch {epoch}/{total_epochs}] Attribute Swap Accuracy: {swap_accuracy:.4f}")

        # 定义保存目录
        save_dir = "train_model"
        os.makedirs(save_dir, exist_ok=True)  # 如果目录不存在，则创建

        # g) 保存模型
        if swap_accuracy > best_swap_accuracy:  # 如果当前模型的属性替换效果最佳
            best_swap_accuracy = swap_accuracy
            save_path = os.path.join(save_dir, "best_autoencoder.pth")  # 构造保存路径
            torch.save(autoencoder.state_dict(), save_path)  # 保存模型状态字典
            print(f"Best model saved based on Attribute Swap Accuracy to {save_path}.")

if __name__ == "__main__":
    train()
    writer.close()  # 关闭 TensorBoard SummaryWriter
