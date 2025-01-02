import os
import torch
import yaml
import datetime

from torch import optim, nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

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
classification_criterion = nn.BCELoss()

# 获取当前时间戳并格式化为文件夹名称
current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
log_dir = f'Tensorboard/train_{current_time}' # 动态生成TensorBoard日志目录
writer = SummaryWriter(log_dir=log_dir) # 初始化TensorBoard的SummaryWriter
print(f"TensorBoard logs saving to: {log_dir}")