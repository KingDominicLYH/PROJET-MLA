import os
import cv2
import torch
from tqdm import tqdm

def preprocess_and_save_dataset(img_dir, attr_file, save_dir, img_size):
    """
    Preprocess images and labels, and save them as .pth files for training, validation, and testing.
    The images are not converted to Tensor here, as ToTensor will handle it later.
    :param img_dir: Path to the directory containing images
    :param attr_file: Path to the attribute file
    :param save_dir: Directory to save the processed datasets
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
    labels = torch.tensor(labels, dtype=torch.float32)
    labels = (labels + 1) / 2  # Convert -1 to 0, keep 1 as 1

    # Calculate number of images for each split (85%, 10%, 5%)
    total_images = len(image_ids)
    train_size = int(0.85 * total_images)
    val_size = int(0.1 * total_images)
    test_size = total_images - train_size - val_size

    # Split image_ids and labels
    train_images = image_ids[:train_size]
    train_labels = labels[:train_size]
    val_images = image_ids[train_size:train_size + val_size]
    val_labels = labels[train_size:train_size + val_size]
    test_images = image_ids[train_size + val_size:]
    test_labels = labels[train_size + val_size:]

    # Process and save training dataset
    print(f"Processing and saving {len(train_images)} training images...")
    process_and_save_images(img_dir, train_images, train_labels, save_dir, "train", img_size)

    # Process and save validation dataset
    print(f"Processing and saving {len(val_images)} validation images...")
    process_and_save_images(img_dir, val_images, val_labels, save_dir, "val", img_size)

    # Process and save testing dataset
    print(f"Processing and saving {len(test_images)} testing images...")
    process_and_save_images(img_dir, test_images, test_labels, save_dir, "test", img_size)

    print("All datasets processed and saved.")

def process_and_save_images(img_dir, image_ids, labels, save_dir, data_type, img_size):
    """
    Process images and save them to .pth file.
    The images are not converted to Tensor here, so they remain as NumPy arrays.
    :param img_dir: Path to the image directory
    :param image_ids: List of image file names
    :param labels: Corresponding labels for the images
    :param save_dir: Directory to save the processed data
    :param data_type: Type of data ('train', 'val', 'test')
    :param img_size: Target size for resized images
    """
    images = []
    for idx, img_name in tqdm(enumerate(image_ids), desc=f"Processing {data_type} images", total=len(image_ids), unit="image"):
        img_path = os.path.join(img_dir, img_name)
        image = cv2.imread(img_path)
        assert image is not None, f"Image not found: {img_path}"

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Crop and resize (Crop the central region)
        image = image[20:-20, :, :]  # Crop the central region
        image = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_LANCZOS4)

        # Append the image (still as NumPy array)
        images.append(image)

    # Save the processed images and labels as a .pth file
    save_path = os.path.join(save_dir, f"{data_type}_dataset.pth")
    data = {"images": images, "labels": labels}
    torch.save(data, save_path)
    print(f"{data_type} dataset saved at {save_path}")







