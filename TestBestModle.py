import torch
from src.models import Classifier
from src.tools import CelebADataset, Config
import random
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import yaml


# ========================
# Configuration and Initialization
# ========================
# Load configuration file
config_file = "parameter/parameters_test.yaml"
with open(config_file, "r") as f:
    params_dict = yaml.safe_load(f)

# Load device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

params = Config(params_dict)

# Initialize the classifier model
model = Classifier(params).to(device)
model_path = "classifier_model/best_model.pth"
model.load_state_dict(torch.load(model_path, map_location=device))

# Path to test dataset
test_data_path = params.preprocess_save_directory + "/test_dataset.pth"

# Attribute names (retrieved from config)
attribute_names = params.target_attribute_list  # List of target attribute names


# ========================
# Load Test Data
# ========================
# Load data from test_dataset.pth file
test_data = torch.load(test_data_path)  # Contains images and labels
images, labels = test_data["images"], test_data["labels"]

# Randomly select 20 images and their corresponding labels
indices = random.sample(range(len(images)), 20)
sampled_images = images[indices]
sampled_labels = labels[indices]

# ========================
# Model Predictions
# ========================
# Move images and labels to the device
sampled_images = sampled_images.to(device)
sampled_labels = sampled_labels.to(device)

# Perform model predictions
outputs = model(sampled_images)  # Model output (batch_size, n_attributes, num_classes)
predictions = torch.argmax(outputs, dim=-1)  # Get predicted class for each attribute
true_labels = torch.argmax(sampled_labels, dim=-1)  # Convert one-hot encoded labels to integers

# ========================
# Accuracy Calculation
# ========================
# Compute accuracy for each attribute
attribute_accuracies = []
for attr_idx, attr_name in enumerate(attribute_names):
    acc = accuracy_score(true_labels[:, attr_idx].cpu(), predictions[:, attr_idx].cpu())
    attribute_accuracies.append((attr_name, acc))
    print(f"Accuracy for {attr_name}: {acc:.4f}")

# Compute mean accuracy across attributes
mean_accuracy = sum([acc for _, acc in attribute_accuracies]) / len(attribute_accuracies)
print(f"Mean Accuracy: {mean_accuracy:.4f}")

# ========================
# Visualization of Predictions
# ========================
# Visualize 20 randomly selected images along with their predictions
for i in range(20):
    image = sampled_images[i].permute(1, 2, 0).cpu().numpy()  # Convert to HWC format
    predicted_attrs = [attribute_names[j] for j in range(len(attribute_names)) if predictions[i, j] == 1]
    true_attrs = [attribute_names[j] for j in range(len(attribute_names)) if true_labels[i, j] == 1]

    # Display image and predictions
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Predicted: {', '.join(predicted_attrs)}\nTrue: {', '.join(true_attrs)}")
    plt.show()
