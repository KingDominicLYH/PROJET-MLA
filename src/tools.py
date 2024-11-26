import torch
from torch.utils.data import Dataset

class CelebADataset(Dataset):
    def __init__(self, processed_file, split="train", transform=None, params=None):
        """
        Initialize the dataset
        :param processed_file: Path to the saved processed dataset
        :param split: Which split to load ("train", "val", "test")
        :param transform: Optional image transformations
        :param params: Parameters dictionary passed from the main program
        """
        print(f"Loading processed dataset from {processed_file}...")
        data = torch.load(processed_file)
        self.images = data["images"]

        # Get the list of attributes to consider
        attribute_list = params["attribute_list"]  # 直接使用简单列表

        all_attr = params["ALL_ATTR"]

        # Filter labels to only include the specified attributes
        if all_attr and attribute_list:
            attr_indices = [all_attr.index(attr) for attr in attribute_list]
            self.labels = data["labels"][:, attr_indices]
        else:
            raise ValueError("Attribute list or ALL_ATTR is missing in params")

        self.labels = self.one_hot_encode(labels=self.labels)

        self.transform = transform

        # Split dataset dynamically
        total_samples = len(self.images)
        train_end = int(0.85 * total_samples)  # 85% for training
        val_end = int(0.95 * total_samples)  # 10% for validation, 5% for testing

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

    @staticmethod
    def one_hot_encode(labels, num_classes=2):
        """
        Perform one-hot encoding for labels and return a 3D tensor.
        :param labels: Input label tensor of shape (N, num_attributes).
        :param num_classes: Number of classes per attribute (default: 2 for binary).
        :return: One-hot encoded 3D tensor of shape (N, num_attributes, num_classes).
        """
        batch_size, num_attrs = labels.size()
        one_hot_labels = torch.zeros(batch_size, num_attrs, num_classes, dtype=torch.float32)
        one_hot_labels.scatter_(-1, labels.long().unsqueeze(-1), 1.0)
        return one_hot_labels