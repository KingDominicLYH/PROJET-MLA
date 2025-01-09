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
params = Config(params_dict)

# Load classifier configuration parameters from the YAML file
with open("parameter/parameters_classifier.yaml", "r") as f:
    params_dict_classifier = yaml.safe_load(f)
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

# Get the current timestamp and format it as a folder name
current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
log_dir = f'Tensorboard/train_{current_time}'  # Dynamically generate TensorBoard log directory

# Check and create directory
os.makedirs(log_dir, exist_ok=True)  # Create directory if it doesn't exist
print(f"TensorBoard logs saving to: {log_dir}")

# Initialize TensorBoard SummaryWriter
writer = SummaryWriter(log_dir=log_dir)

os.makedirs(params.model_output_path, exist_ok=True)

def train():
    best_swap_accuracy = 0.0
    save_start_epoch = 150  # Start saving models and applying early stopping from this epoch
    no_improvement_count = 0  # Counter for consecutive epochs without improvement
    patience = 50  # Stop training if validation accuracy does not improve for 50 epochs

    # Compute number of iterations per epoch
    num_iterations = params.total_train_samples // params.batch_size
    step_counter = 0  # Track the current training step

    total_epochs = params.total_epochs
    for epoch in range(1, total_epochs + 1):
        print(f'Starting Epoch {epoch}/{total_epochs}')
        autoencoder.train()
        discriminator.train()

        # Initialize tracking variables
        total_recon_loss = 0.0
        total_discriminator_loss = 0.0
        total_adversarial_loss = 0.0
        num_samples = 0

        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs}", dynamic_ncols=True, total=num_iterations)
        for step, (images, labels) in enumerate(train_loader_tqdm):
            if step >= num_iterations:
                break  # Stop training after reaching 50000 samples
            label_indices = labels.argmax(dim=-1)  # [N, 40, 2] -> [N, 40]
            images, labels, label_indices = images.to(device), labels.to(device), label_indices.to(device)

            # Dynamically adjust discriminator_loss_weight
            max_steps = 500000
            params.latent_discriminator_loss_weight = min(
                0.001,  # Maximum value
                (step_counter / max_steps) * 0.001  # Gradually increase
            )

            # (i) Train discriminator
            latent = autoencoder.encoder(images)
            pred_labels_real = discriminator(latent.detach())
            loss_discriminator = classification_criterion(pred_labels_real.permute(0, 2, 1), label_indices)

            discriminator_optimizer.zero_grad()
            loss_discriminator.backward()
            discriminator_optimizer.step()

            # (ii) Train autoencoder
            latent = autoencoder.encoder(images)
            reconstructed_images = autoencoder.decoder(latent, labels)
            recon_loss = reconstruction_criterion(reconstructed_images, images)

            # Flip labels => 1 - labels
            pred_labels_fake = discriminator(latent)
            adversarial_loss = classification_criterion(pred_labels_fake.permute(0, 2, 1), 1 - label_indices)

            ae_loss = params.autoencoder_loss_weight * recon_loss + params.latent_discriminator_loss_weight * adversarial_loss

            autoencoder_optimizer.zero_grad()
            ae_loss.backward()
            autoencoder_optimizer.step()

            # Accumulate losses
            total_recon_loss += recon_loss.item() * params.batch_size
            total_discriminator_loss += loss_discriminator.item() * params.batch_size
            total_adversarial_loss += adversarial_loss.item() * params.batch_size
            num_samples += params.batch_size

            # Log and update progress bar
            train_loader_tqdm.set_postfix({
                "Recon": recon_loss.item(),
                "D_loss": loss_discriminator.item(),
                "Adv": adversarial_loss.item()
            })

            step_counter += params.batch_size

        # Compute average losses per epoch
        avg_recon_loss = total_recon_loss / num_samples
        avg_discriminator_loss = total_discriminator_loss / num_samples
        avg_adversarial_loss = total_adversarial_loss / num_samples

        # Log metrics to TensorBoard
        writer.add_scalar("Epoch_Avg_Reconstruction_Loss", avg_recon_loss, epoch)
        writer.add_scalar("Epoch_Avg_Discriminator_Loss", avg_discriminator_loss, epoch)
        writer.add_scalar("Epoch_Avg_Adversarial_Loss", avg_adversarial_loss, epoch)

        # Validate model
        autoencoder.eval()
        classifier.eval()

        swap_accuracy = 0.0
        num_samples = 0

        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)

                # Modify attributes (e.g., toggle glasses attribute)
                modified_labels = labels.clone()
                modified_labels[:, :, 0], modified_labels[:, :, 1] = modified_labels[:, :, 1], modified_labels[:, :, 0].clone()

                swapped_images = autoencoder(images, modified_labels)
                modified_labels = modified_labels.argmax(dim=2)

                predicted_labels = classifier(swapped_images).argmax(dim=2)
                target_predicted_labels = predicted_labels[:, target_attribute_indices]

                swap_accuracy += (target_predicted_labels == modified_labels).sum().item()
                num_samples += images.size(0)

        swap_accuracy /= num_samples
        writer.add_scalar("Validation/Attribute_Swap_Accuracy", swap_accuracy, epoch)

        print(f"[Epoch {epoch}/{total_epochs}] Attribute Swap Accuracy: {swap_accuracy:.4f}")

        os.makedirs(params.save_dir, exist_ok=True)

        # Save model
        if epoch <= save_start_epoch:
            continue
        else:
            if swap_accuracy > best_swap_accuracy:
                best_swap_accuracy = swap_accuracy
                attributes_str = "_".join(map(str, params.target_attribute_list))
                save_path = os.path.join(params.save_dir, f"best_autoencoder_{attributes_str}.pth")

                torch.save(autoencoder.state_dict(), save_path)
                print(f"Best model saved based on Attribute Swap Accuracy to {save_path}.")
            else:
                no_improvement_count += 1
                print(f"[Epoch {epoch + 1}] No improvement. Count {no_improvement_count}/{patience}.")

                if no_improvement_count >= patience:
                    print("Early stopping triggered!")
                    break

if __name__ == "__main__":
    train()
    writer.close()
