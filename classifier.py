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

# Load YAML configuration
with open("parameter/parameters_classifier.yaml", "r") as f:
    params_dict = yaml.safe_load(f)
params = Config(params_dict)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data loading
train_dataset = CelebADataset(data_dir=params.processed_file, params=params, split="train")
valid_dataset = CelebADataset(data_dir=params.processed_file, params=params, split="val")

train_loader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=params.batch_size, shuffle=False, pin_memory=True)

# Model, loss function, and optimizer
model = Classifier(params).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = get_optimizer(model, params.optimizer)

# Get the current timestamp and format it as a folder name
current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
log_dir = f'Tensorboard/{current_time}'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
    print(f"Directory {log_dir} created.")
writer = SummaryWriter(log_dir=log_dir)


def train(model, train_loader, valid_loader, criterion, optimizer, n_epochs, device):
    """
    Training function:
    - First 100 epochs do not save the model.
    - From epoch 101 onwards, Early-Stopping is applied (based on validation accuracy).
    """

    # ====== Variables for Early-Stopping ======
    best_valid_accuracy = 0.0  # Track the best validation accuracy
    patience = 50  # Stop training if validation accuracy does not improve for 50 epochs
    no_improvement_count = 0  # Counter for epochs with no improvement
    save_start_epoch = 150  # Start saving models and applying Early-Stopping after epoch 150

    # Compute the number of iterations per epoch
    num_iterations = params.total_train_samples // params.batch_size
    num_valid_iterations = params.total_valid_samples // params.batch_size

    for epoch in range(n_epochs):
        print(f'Starting Epoch {epoch + 1}/{n_epochs}')

        # ====== Training Phase ======
        model.train()
        train_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        num_samples = 0

        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{n_epochs}",
                                 dynamic_ncols=True, total=num_iterations)

        for i, (inputs, labels) in enumerate(train_loader_tqdm):
            if i >= num_iterations:
                break  # Stop training after reaching the specified number of training samples
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

        # ====== Validation Phase ======
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

        # ====== Print training and validation metrics ======
        print(f'Epoch {epoch + 1}/{n_epochs}')
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
        print(f'Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.4f}')

        # Write metrics to TensorBoard
        writer.add_scalar('Train Loss', train_loss, epoch)
        writer.add_scalar('Train Accuracy', train_accuracy, epoch)
        writer.add_scalar('Valid Loss', valid_loss, epoch)
        writer.add_scalar('Valid Accuracy', valid_accuracy, epoch)

        # ====== Ensure the save directory exists ======
        if not os.path.exists(params.save_dir):
            os.makedirs(params.save_dir)
            print(f"Directory {params.save_dir} created.")

        # ====== Early-Stopping logic (starting from epoch 101) ======
        if epoch + 1 <= save_start_epoch:
            # Do not save models or apply Early-Stopping in the first 100 epochs
            continue
        else:
            # Save the model only if validation accuracy improves
            if valid_accuracy > best_valid_accuracy:
                best_valid_accuracy = valid_accuracy
                no_improvement_count = 0
                save_path = os.path.join(params.save_dir, f"best_model_epoch_{epoch+1}.pth")
                torch.save(model.state_dict(), save_path)
                print(f"[Epoch {epoch + 1}] Model saved to {save_path} with valid acc {valid_accuracy:.4f}!")
            else:
                # If validation accuracy does not improve, increase the counter
                no_improvement_count += 1
                print(f"[Epoch {epoch + 1}] No improvement. Count {no_improvement_count}/{patience}.")

                # Stop training if no improvement for 50 consecutive epochs
                if no_improvement_count >= patience:
                    print("Early stopping triggered!")
                    break


# Run training
train(model, train_loader, valid_loader, criterion, optimizer, params.total_epochs, device)
