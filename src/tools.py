import torch
from torch.utils.data import Dataset


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