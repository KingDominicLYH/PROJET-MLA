import os
import torch
import yaml
import datetime

from torch import optim, nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.models import AutoEncoder, Discriminator
from src.tools import CelebADataset, Config

# Load configuration parameters from the YAML file
with open("parameter/parameters.yaml", "r") as f:
    params_dict = yaml.safe_load(f)

# 将YAML配置字典转换为Config对象
params = Config(params_dict)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data preparation: load training and validation datasets
train_dataset = CelebADataset(data_dir=params.processed_file, params=params, split="train")
valid_dataset = CelebADataset(data_dir=params.processed_file, params=params, split="val")

train_loader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=params.batch_size, shuffle=False, pin_memory=True)

print("Train dataset size:", len(train_dataset))
print("Validation dataset size:", len(valid_dataset))
print(f"n_attributes = {params.n_attributes}")

# Initialize models
autoencoder = AutoEncoder(params).to(device)
discriminator = Discriminator(params).to(device)

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
log_dir = f'Tensorboard/train_{current_time}' # 动态生成TensorBoard日志目录
writer = SummaryWriter(log_dir=log_dir) # 初始化TensorBoard的SummaryWriter
print(f"TensorBoard logs saving to: {log_dir}")

os.makedirs(params.model_output_path, exist_ok=True)

def train():
    best_val_loss = float('inf')

    # 计算每个 epoch 中的迭代次数
    num_iterations = params.total_train_samples // params.batch_size
    num_valid_iterations = params.total_valid_samples // params.batch_size

    total_epochs = params.total_epochs
    for epoch in range(1, total_epochs + 1):
        print(f'Starting Epoch {epoch + 1}/{total_epochs}')
        autoencoder.train()
        discriminator.train()

        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs}", dynamic_ncols=True, total=num_iterations)
        for step, (images, labels) in enumerate(train_loader_tqdm):
            if step >= num_iterations:
                break  # 超过50000个样本后，终止训练
            label_indices = labels.argmax(dim=-1)  # [N, 40, 2] -> [N, 40]
            images, labels, label_indices = images.to(device), labels.to(device), label_indices.to(device)

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

            # (iii) 日志 & 进度条
            train_loader_tqdm.set_postfix({
                "Recon": recon_loss.item(),
                "D_loss": loss_discriminator.item(),
                "Adv": adversarial_loss.item()
            })
            global_step = epoch * len(train_loader) + step
            writer.add_scalar("Train/Reconstruction_Loss", recon_loss.item(), global_step)
            writer.add_scalar("Train/Discriminator_Loss", loss_discriminator.item(), global_step)
            writer.add_scalar("Train/Adversarial_Loss", adversarial_loss.item(), global_step)

        # f) 验证
        autoencoder.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                recon_images = autoencoder(images, labels)
                val_loss = reconstruction_criterion(recon_images, images)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(valid_loader)
        writer.add_scalar("Validation/Reconstruction_Loss", avg_val_loss, epoch)
        print(f"[Epoch {epoch}/{total_epochs}] Validation Loss: {avg_val_loss:.6f}")

        # g) 保存模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss