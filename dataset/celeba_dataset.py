import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from tqdm import tqdm  # 用于显示进度条 / For progress bar visualization


def preprocess_and_save_dataset(img_dir, attr_file, save_path, img_size=256):
    """
    Preprocess images and labels, and save them as a .pth file
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

    # Convert labels to tensor
    labels = np.array(labels, dtype=np.float32)
    labels = (labels + 1) / 2  # Convert -1 to 0, keep 1 as 1
    labels = torch.tensor(labels)

    # Process images
    images = []
    print("Processing images...")
    for img_name in tqdm(image_ids, desc="Processing images", unit="image"):
        img_path = os.path.join(img_dir, img_name)
        image = cv2.imread(img_path)
        assert image is not None, f"Image not found: {img_path}"

        # Crop and resize
        image = image[20:-20, :, :]  # Crop the central region
        image = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_AREA)

        # Normalize and convert to tensor
        image = image.astype(np.float32) / 255.0
        image = (image * 2) - 1  # Normalize to [-1, 1]
        image = np.transpose(image, (2, 0, 1))  # Convert to (C, H, W)
        images.append(image)

    # Convert images to a tensor
    images = torch.tensor(np.stack(images))  # Stack all images into a single tensor

    # Save processed data
    print(f"Saving processed dataset to {save_path}...")
    torch.save({"images": images, "labels": labels}, save_path)
    print("Dataset saved!")


class CelebADataset(Dataset):
    def __init__(self, processed_file, split="train", transform=None):
        """
        Initialize the dataset
        :param processed_file: Path to the saved processed dataset
        :param split: Which split to load ("train", "val", "test")
        :param transform: Optional image transformations
        """
        print(f"Loading processed dataset from {processed_file}...")
        data = torch.load(processed_file)
        self.images = data["images"]
        self.labels = data["labels"]
        self.transform = transform

        # Split the dataset
        train_end = 170000
        val_end = train_end + 20000

        if split == "train":
            self.images = self.images[:train_end]
            self.labels = self.labels[:train_end]
        elif split == "val":
            self.images = self.images[train_end:val_end]
            self.labels = self.labels[train_end:val_end]
        elif split == "test":
            self.images = self.images[val_end:]
            self.labels = self.labels[val_end:]
        else:
            raise ValueError("Split must be 'train', 'val', or 'test'")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        Retrieve an image and its label by index
        :param idx: Index
        :return: Image tensor and label
        """
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# if __name__ == "__main__":
#     # Configuration
#     IMG_DIR = "img_align_celeba"  # Image folder
#     ATTR_FILE = "list_attr_celeba.txt"  # Attribute file
#     PROCESSED_FILE = "celeba_dataset.pth"  # Processed dataset file
#     IMG_SIZE = 256  # Image resize dimensions
#
#     # Preprocess and save dataset (only run this once)
#     if not os.path.isfile(PROCESSED_FILE):
#         preprocess_and_save_dataset(IMG_DIR, ATTR_FILE, PROCESSED_FILE, img_size=IMG_SIZE)
#
#     # Load datasets for each split
#     train_dataset = CelebADataset(processed_file=PROCESSED_FILE, split="train")
#     val_dataset = CelebADataset(processed_file=PROCESSED_FILE, split="val")
#     test_dataset = CelebADataset(processed_file=PROCESSED_FILE, split="test")
#
#     # Create DataLoaders
#     train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
#     test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
#
#     # Example usage
#     for images, labels in train_loader:
#         print(f"Train batch - images shape: {images.shape}, labels shape: {labels.shape}")
#         break
#
#     for images, labels in val_loader:
#         print(f"Validation batch - images shape: {images.shape}, labels shape: {labels.shape}")
#         break
#
#     for images, labels in test_loader:
#         print(f"Test batch - images shape: {images.shape}, labels shape: {labels.shape}")
#         break

# 仅用来测试 preprocess_and_save_dataset 函数
if __name__ == "__main__":
    # Configuration
    IMG_DIR = "img_align_celeba"  # Image folder
    ATTR_FILE = "list_attr_celeba.txt"  # Attribute file
    PROCESSED_FILE = "celeba_dataset.pth"  # Processed dataset file
    IMG_SIZE = 256  # Image resize dimensions

    # Test preprocess_and_save_dataset
    if not os.path.isfile(PROCESSED_FILE):
        print("Preprocessing and saving dataset...")
        preprocess_and_save_dataset(IMG_DIR, ATTR_FILE, PROCESSED_FILE, img_size=IMG_SIZE)

    # Check if the file was saved correctly
    if os.path.isfile(PROCESSED_FILE):
        print(f"File {PROCESSED_FILE} created successfully.")

        # Load the saved file to verify
        data = torch.load(PROCESSED_FILE)
        images = data["images"]
        labels = data["labels"]

        print(f"Dataset contains {images.size(0)} images and {labels.size(0)} labels.")
        print("Displaying the first few samples...")

        # Display first few samples
        for i in range(5):
            print(f"\nSample {i + 1}:")
            print(f"Image shape: {images[i].shape}")
            print(f"Label: {labels[i]}")
            print(f"Pixel range: min={images[i].min()}, max={images[i].max()}")  # Check normalization
    else:
        print("File saving failed!")

