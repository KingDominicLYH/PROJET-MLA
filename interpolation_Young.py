import os
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt

# Import Config (if applicable) and AutoEncoder based on your project structure
from src.tools import Config  # Ensure this file and class exist
from src.models import AutoEncoder

# ======== Set the attributes of interest and list of original images ========

raw_imgs = ["008507.jpg"]  # You can add more image names, e.g., ["008507.jpg", "000008.jpg", ...]

# Load training parameters from the YAML file to ensure consistency
with open("parameter/parameters.yaml", "r") as f:
    params_dict = yaml.safe_load(f)

params = Config(params_dict)  # Convert YAML config dictionary into a Config object
params.target_attribute_list = ["Young"]

# 1. Load trained model function
def load_trained_autoencoder(model_path: str, device: torch.device, params):
    """
    Load a trained AutoEncoder model.
    """
    # Load model weights from file
    checkpoint = torch.load(model_path, map_location=device)

    # Initialize model and load weights
    model = AutoEncoder(params).to(device)
    model.load_state_dict(checkpoint)
    model.eval()

    return model

# 2. Preprocess a single image
def load_and_preprocess_image(img_path: str, img_size: int = 256):
    """
    Load and preprocess a single image:
      - Convert BGR to RGB
      - Center crop (if done during training)
      - Resize to (img_size, img_size)
      - Normalize to the range [-1, 1] as a tensor
    """
    # Read the image (BGR format)
    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # If training involved cropping 20 pixels from the top and bottom, ensure consistency
    if image.shape[0] > 40:
        image = image[20:-20, :, :]

    # Resize to target size
    image = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_LANCZOS4)

    # Convert to tensor and normalize to [-1, 1]
    image = torch.from_numpy(image.transpose((2, 0, 1))).float()  # [3, H, W]
    image = image / 255.0  # Normalize to [0,1]
    image = image * 2.0 - 1.0  # Normalize to [-1,1]

    # Add batch dimension [1, 3, H, W]
    image = image.unsqueeze(0)
    return image

# 3. Core function for attribute interpolation
def generate_attribute_interpolations(
    model: AutoEncoder,
    input_image: torch.Tensor,
    device: torch.device,
    n_interpolations: int = 10
):
    """
    Perform attribute interpolation on a single image and return (n_interpolations) generated images.
    Assuming only one attribute (n_attributes=1), the attribute vector shape is [1, 2].
    Example: Linear interpolation from [1,0] to [0,1].
    """
    input_image = input_image.to(device)

    # Extract latent representation of the image
    with torch.no_grad():
        latent = model.encoder(input_image)  # [1, 512, h', w']

    # Alpha values from 0 to 1
    alphas = np.linspace(0, 1, n_interpolations)

    output_images = []
    with torch.no_grad():
        for alpha in alphas:
            # Single attribute interpolation: attribute = [1 - alpha, alpha]
            attribute = torch.tensor([[1.0 - alpha, alpha]], device=device, dtype=torch.float32)  # Ensure float32
            # Decode
            out = model.decoder(latent, attribute)
            output_images.append(out.cpu())

    output_images = torch.cat(output_images, dim=0)  # [n_interpolations, 3, H, W]
    return output_images

# 4. Main logic: Perform interpolation for each image in the list
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load trained model (update path as needed)
    model_path = "train_model/best_autoencoder_Young.pth"
    autoencoder = load_trained_autoencoder(model_path, device, params)

    # Process each image one by one
    n_interpolations = 10  # Number of interpolation points
    for img_name in raw_imgs:
        # Construct the image path (modify if needed)
        test_img_path = r"dataset\img_align_celeba\008507.jpg"
        # test_img_path = os.path.join(params.raw_img_directory, img_name)

        # Check if the image path is valid
        if not os.path.isfile(test_img_path):
            print(f"Image file not found: {test_img_path}")
            continue

        # Preprocess image
        input_image = load_and_preprocess_image(test_img_path, img_size=256)

        # Perform interpolation
        output_images = generate_attribute_interpolations(
            model=autoencoder,
            input_image=input_image,
            device=device,
            n_interpolations=n_interpolations
        )

        # Generate image grid
        # Arrange images in a single row => nrow=n_interpolations
        # (5) Visualization: Create an image grid and save the result
        # Normalize output_images to [0, 1] for visualization
        output_images = (output_images - output_images.min()) / (output_images.max() - output_images.min())

        # Create an image grid without using range parameter
        grid = make_grid(output_images, nrow=n_interpolations)

        # Save result
        save_filename = f"interpolation_{os.path.splitext(img_name)[0]}.png"
        save_image(grid, save_filename)
        print(f"Interpolation result of {img_name} saved to: {save_filename}")

        # Optionally visualize directly using matplotlib
        grid_np = grid.permute(1, 2, 0).cpu().numpy()  # Convert from C,H,W to H,W,C
        plt.figure(figsize=(20, 4))  # Make it wider
        plt.imshow(grid_np)
        plt.title(f"Interpolations of {img_name}")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
