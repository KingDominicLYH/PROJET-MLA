import os

import torch
import yaml
from torch import optim, nn
from torch.utils.data import DataLoader

from src.models import AutoEncoder, Discriminator
from src.tools import CelebADataset

# Load configuration parameters from the YAML file
with open("parameter/parameters.yaml", "r") as f:
    params = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data preparation: load training and validation datasets
train_dataset = CelebADataset(data_dir=params.processed_file, params=params, split="train")
valid_dataset = CelebADataset(data_dir=params.processed_file, params=params, split="val")

train_loader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=params.batch_size, shuffle=False, pin_memory=True)

# Initialize models
autoencoder = AutoEncoder(params.n_attributes).to(device)
discriminator = Discriminator(params.n_attributes).to(device)

# Setup optimizers
autoencoder_optimizer = optim.Adam(autoencoder.parameters(), lr=params.learning_rate)
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=params.learning_rate)

# Define loss functions
reconstruction_criterion = nn.MSELoss()
classification_criterion = nn.BCELoss()

# Main training loop
os.makedirs(model_output_path, exist_ok=True)
best_val_loss = float("inf")