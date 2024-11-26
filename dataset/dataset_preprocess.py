import os
import numpy as np
import torch
import cv2
from tqdm import tqdm


def preprocess_and_save_dataset(img_dir, attr_file, save_path, img_size=256):
    """
    Preprocess images and labels, and save them as a .pth file.
    :param img_dir: Path to the directory containing images
    :param attr_file: Path to the attribute file
    :param save_path: Path to save the processed data
    :param img_size: Target size for resized images
    """
    print("Processing and saving dataset...")

    # Read attribute file
    print("Reading attribute file...")
    with open(attr_file, 'r') as f:
        attr_lines = f.readlines()
    attr_keys = attr_lines[1].split()  # Attribute names
    labels = []
    image_ids = []
    for line in attr_lines[2:]:
        parts = line.split()
        image_ids.append(parts[0])  # Image file name
        labels.append([int(x) for x in parts[1:]])  # Attributes

    # Convert labels to NumPy array
    labels = np.array(labels, dtype=np.float32)
    labels = (labels + 1) / 2  # Convert -1 to 0, keep 1 as 1

    # Process all images
    print(f"Processing {len(image_ids)} images...")
    all_images = []
    for img_name in tqdm(image_ids, desc="Processing images", unit="image"):
        img_path = os.path.join(img_dir, img_name)
        image = cv2.imread(img_path)
        assert image is not None, f"Image not found: {img_path}"

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Crop and resize
        image = image[20:-20, :, :]  # Crop the central region
        image = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_LANCZOS4)

        # Normalize to [-1, 1]
        # image = image.astype(np.float32) / 255.0
        # image = (image * 2) - 1  # Normalize to [-1, 1]
        all_images.append(image)

    # Convert to PyTorch tensors
    print("Converting data to tensors...")
    all_tensor_imgs = np.concatenate([img.transpose((2, 0, 1))[None] for img in all_images], 0)
    all_tensor_imgs = torch.from_numpy(all_tensor_imgs)
    assert all_tensor_imgs.size() == (202599, 3, IMG_SIZE, IMG_SIZE)
    all_labels = torch.tensor(labels)  # Convert labels to tensor

    # Save processed data
    print(f"Saving processed dataset to {save_path}...")
    torch.save({"images": all_tensor_imgs, "labels": all_labels}, save_path)
    print("Dataset saved!")

def normalize_and_save_dataset(processed_file, save_path):
    """
    Normalize the images in a processed dataset and save the normalized dataset.
    :param processed_file: Path to the processed dataset file
    :param save_path: Path to save the normalized dataset
    """
    print(f"Loading processed dataset from {processed_file}...")
    data = torch.load(processed_file)
    images = data["images"]  # Shape: (N, 3, H, W)
    labels = data["labels"]  # Shape: (N, num_attributes)

    print("Normalizing images...")
    # Normalize images to [-1, 1]
    images = images.float() / 255.0  # Convert to float and normalize to [0, 1]
    images = images * 2 - 1  # Normalize to [-1, 1]

    # Save normalized dataset
    print(f"Saving normalized dataset to {save_path}...")
    torch.save({"images": images, "labels": labels}, save_path)
    print("Normalized dataset saved!")




if __name__ == "__main__":
    # Configuration
    IMG_DIR = "img_align_celeba"  # Image folder
    ATTR_FILE = "list_attr_celeba.txt"  # Attribute file
    PROCESSED_FILE = "celeba_dataset.pth"  # Original processed dataset file
    NORMALIZED_FILE = "celeba_normalized_dataset.pth"  # File to save normalized dataset
    IMG_SIZE = 256  # Image resize dimensions

    # Preprocess and save dataset
    if not os.path.isfile(PROCESSED_FILE):
        print("Preprocessing and saving dataset...")
        preprocess_and_save_dataset(IMG_DIR, ATTR_FILE, PROCESSED_FILE, img_size=IMG_SIZE)

    # Normalize and save dataset
    if not os.path.isfile(NORMALIZED_FILE):
        print("Normalizing and saving dataset...")
        normalize_and_save_dataset(PROCESSED_FILE, NORMALIZED_FILE)

    # Check if the normalized file was saved correctly
    if os.path.isfile(NORMALIZED_FILE):
        print(f"File {NORMALIZED_FILE} created successfully.")

        # Load the normalized file to verify
        data = torch.load(NORMALIZED_FILE)
        images = data["images"]
        labels = data["labels"]

        print(f"Normalized dataset contains {images.size(0)} images and {labels.size(0)} labels.")
        print("Displaying the first few normalized samples...")

        # Display first few samples
        for i in range(5):
            print(f"\nSample {i + 1}:")
            print(f"Image shape: {images[i].shape}")
            print(f"Label: {labels[i]}")
            print(f"Pixel range: min={images[i].min().item()}, max={images[i].max().item()}")  # Check normalization
    else:
        print("File saving failed!")

