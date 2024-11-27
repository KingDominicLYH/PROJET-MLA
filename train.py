import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
from src.tools import CelebADataset
from src.models import AutoEncoder, Discriminator
import yaml

# Load configuration parameters from the YAML file
with open("parametre/parameters.yaml", "r") as f:
    params = yaml.safe_load(f)

# Hyperparameter setup
batch_size = params["batch_size"]  # Number of samples per batch
image_size = params["image_size"]  # Input image size (height and width)
latent_discriminator_loss_weight = params["latent_discriminator_loss_weight"]  # Weight for adversarial loss
autoencoder_loss_weight = params["autoencoder_loss_weight"]  # Weight for reconstruction loss
epochs = params["total_epochs"]  # Total number of training epochs
lr = float(params["autoencoder_optimizer"].split(",lr=")[1])  # Learning rate for optimizers
latent_discriminator_steps = params["latent_discriminator_steps"]  # Number of discriminator updates per AE update
model_output_path = params["model_output_path"]  # Directory to save model checkpoints
n_attributes = params["attribute_count"]  # Number of attributes in the dataset
subset_train_size = params["samples_train_per_epoch"]  # Number of samples per epoch
subset_val_size = params["samples_val_per_epoch"]  # Number of validation samples
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available

# 每轮训练动态采样
def get_random_subset_loader(dataset, batch_size, subset_size):
    """
    Creates a DataLoader with a random subset of the dataset.
    """
    # 随机生成子集索引
    indices = torch.randperm(len(dataset))[:subset_size]
    sampler = SubsetRandomSampler(indices)

    # 创建 DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=4)
    return loader

# Data preparation: load training and validation datasets
train_dataset = CelebADataset(
    "celeba_normalized_dataset.pth",
    split="train",
    enable_flip=True,
    params=params
)

val_dataset = CelebADataset(
    "celeba_normalized_dataset.pth",
    split="val",
    enable_flip=False,
    params=params
)
val_loader = get_random_subset_loader(val_dataset, batch_size, subset_val_size)

# Initialize the AutoEncoder and Discriminator models
autoencoder = AutoEncoder(n_attributes).to(device)
discriminator = Discriminator(n_attributes).to(device)

# Setup optimizers for AutoEncoder and Discriminator
autoencoder_optimizer = optim.Adam(autoencoder.parameters(), lr=lr, betas=(0.5, 0.999))
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# Define the loss functions
reconstruction_criterion = nn.MSELoss()  # Mean Squared Error for reconstruction loss
classification_criterion = nn.BCELoss()  # Binary Cross Entropy for classification loss

# Training function for a single epoch
def train_one_epoch(epoch, train_loader, autoencoder, discriminator, autoencoder_optimizer, discriminator_optimizer):
    """
    Train the AutoEncoder and Discriminator for one epoch.
    Args:
        epoch (int): Current epoch number.
        train_loader (DataLoader): DataLoader for the training set.
        autoencoder (nn.Module): AutoEncoder model.
        discriminator (nn.Module): Discriminator model.
        autoencoder_optimizer (Optimizer): Optimizer for the AutoEncoder.
        discriminator_optimizer (Optimizer): Optimizer for the Discriminator.
    """
    autoencoder.train()  # Set AutoEncoder to training mode
    discriminator.train()  # Set Discriminator to training mode
    loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")  # Progress bar for the epoch

    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        # --- Train the Discriminator ---
        latent = autoencoder.encoder(images)  # Encode the images to latent space
        pred_labels = discriminator(latent.detach())  # Discriminator predicts attributes from latent space
        discriminator_loss = classification_criterion(pred_labels, labels)  # Calculate classification loss

        discriminator_optimizer.zero_grad()  # Zero the gradients of the Discriminator
        discriminator_loss.backward()  # Backpropagate the loss
        discriminator_optimizer.step()  # Update Discriminator parameters

        # --- Train the AutoEncoder ---
        latent = autoencoder.encoder(images)  # Encode the images again
        reconstructed_images = autoencoder.decoder(latent, labels)  # Decode using true attributes

        # Calculate reconstruction loss
        reconstruction_loss = reconstruction_criterion(reconstructed_images, images)

        # Adversarial loss to fool the Discriminator
        pred_labels = discriminator(latent)  # Predictions from the Discriminator
        adversarial_loss = classification_criterion(pred_labels, 1 - labels)  # Encourage incorrect predictions

        # Combine reconstruction and adversarial losses
        autoencoder_loss = (
            autoencoder_loss_weight * reconstruction_loss
            + latent_discriminator_loss_weight * adversarial_loss
        )

        autoencoder_optimizer.zero_grad()  # Zero the gradients of the AutoEncoder
        autoencoder_loss.backward()  # Backpropagate the loss
        autoencoder_optimizer.step()  # Update AutoEncoder parameters

        # Update progress bar with current losses
        loop.set_postfix({
            "Reconstruction Loss": reconstruction_loss.item(),
            "Discriminator Loss": discriminator_loss.item(),
            "Adversarial Loss": adversarial_loss.item(),
        })

# Validation function to evaluate the AutoEncoder
def validate(val_loader, autoencoder):
    """
    Validate the AutoEncoder on the validation set.
    Args:
        val_loader (DataLoader): DataLoader for the validation set.
        autoencoder (nn.Module): AutoEncoder model.
    Returns:
        float: Average reconstruction loss on the validation set.
    """
    autoencoder.eval()  # Set AutoEncoder to evaluation mode
    total_loss = 0
    with torch.no_grad():  # No gradient computation during validation
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            reconstructed_images = autoencoder(images, labels)  # Reconstruct images
            loss = reconstruction_criterion(reconstructed_images, images)  # Calculate reconstruction loss
            total_loss += loss.item()
    return total_loss / len(val_loader)  # Return average loss

# Main training script
def main():
    """
    Main function to train the AutoEncoder and Discriminator.
    """
    os.makedirs(model_output_path, exist_ok=True)  # Create directory for saving models
    best_val_loss = float("inf")  # Track the best validation loss

    for epoch in range(1, epochs + 1):
        # Train for one epoch
        train_loader = get_random_subset_loader(train_dataset, batch_size, subset_train_size)

        train_one_epoch(epoch, train_loader, autoencoder, discriminator, autoencoder_optimizer, discriminator_optimizer)

        # Validate the AutoEncoder
        val_loss = validate(val_loader, autoencoder)
        print(f"Epoch {epoch}: Validation Loss = {val_loss}")

        # Save the best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(autoencoder.state_dict(), os.path.join(model_output_path, "best_autoencoder.pth"))
            print("Best model saved!")

        # Periodically save model checkpoints
        if epoch % 10 == 0:
            torch.save(autoencoder.state_dict(), os.path.join(model_output_path, f"autoencoder_epoch_{epoch}.pth"))
            torch.save(discriminator.state_dict(), os.path.join(model_output_path, f"discriminator_epoch_{epoch}.pth"))

