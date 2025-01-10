import os
import random
import torch
from torch.utils.data import Dataset
from torch import optim
import re
import inspect
from torchvision import transforms


class CelebADataset(Dataset):
    def __init__(self, data_dir, params, split="train", enable_flip=False, transform=None):
        """
        Initialize the dataset
        :param processed_file: Path to the saved processed dataset
        :param split: Which split to load ("train", "val", "test")
        :param enable_flip: Boolean, whether to enable random horizontal flipping
        :param transform: Optional image transformations
        :param params: Parameters dictionary passed from the main program
        """
        # Sélection dynamique du fichier traité en fonction du split
        processed_file = self._get_processed_file(data_dir, split)
        print(f"Loading processed dataset from {processed_file}...")

        data = torch.load(processed_file)
        self.images = data["images"]

        # Get the list of attributes to consider
        attribute_list = params.target_attribute_list  # Utiliser directement une liste simple

        all_attr = params.ALL_ATTR

        # Filter labels to only include the specified attributes
        if attribute_list == "ALL":
            self.labels = data["labels"]
        else:
            if all_attr and attribute_list:
                attr_indices = [all_attr.index(attr) for attr in attribute_list]
                self.labels = data["labels"][:, attr_indices]
            else:
                raise ValueError("Attribute list or ALL_ATTR is missing in params")

        self.transform = transform
        self.enable_flip = enable_flip  # Paramètre ajouté, utilisé pour contrôler l'activation de la symétrie horizontale

    def _get_processed_file(self, base_dir, split):
        """
        Retourne le chemin du fichier traité en fonction du split
        :param base_dir: Chemin de base où sont stockées les données traitées
        :param split: Fraction du jeu de données (train, val, test)
        :return: Chemin du fichier correspondant
        """
        file_map = {
            "train": "train_dataset.pth",
            "val": "val_dataset.pth",
            "test": "test_dataset.pth"
        }
        if split not in file_map:
            raise ValueError("Invalid split. Expected one of ['train', 'val', 'test'].")

        return os.path.join(base_dir, file_map[split])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        Retrieve an image and its label by index
        :param idx:
        :return: Image tensor and label
        """
        image = self.images[idx]
        label = self.labels[idx]

        # Assurer que l'image est de type float et normalisée dans l'intervalle [0, 1]
        image = normalize_images(image)

        # Applique une symétrie horizontale avec une probabilité de 50%
        if self.enable_flip and random.random() < 0.5:
            image = torch.flip(image, dims=[2])

        if self.transform:
            image = self.transform(image)

        return image, label

    @staticmethod
    def one_hot_encode(labels, num_classes=2):
        """
        Effectue un one-hot encoding des étiquettes et retourne un tenseur 3D.
        :param labels: Tenseur d'étiquettes de forme (N, num_attributes).
        :param num_classes: Nombre de classes par attribut (défaut : 2 pour binaire).
        :return: Tenseur one-hot encodé de forme (N, num_attributes, num_classes).
        """
        batch_size, num_attrs = labels.size()
        one_hot_labels = torch.zeros(batch_size, num_attrs, num_classes, dtype=torch.float32)
        one_hot_labels.scatter_(-1, labels.long().unsqueeze(-1), 1.0)
        return one_hot_labels


class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)


def normalize_images(images):
    """
    Normalise les valeurs d'image en utilisant des opérations en place de manière chaînée.
    """
    return images.float().div(255.0).mul(2.0).add(-1)


def get_optimizer(model, s):
    """
    Analyse les paramètres de l'optimiseur.
    L'entrée doit être sous la forme :
        - "sgd,lr=0.01"
        - "adagrad,lr=0.1,lr_decay=0.05"
    """
    if "," in s:
        method = s[:s.find(',')]
        optim_params = {}
        for x in s[s.find(',') + 1:].split(','):
            split = x.split('=')
            assert len(split) == 2
            assert re.match(r"^[+-]?(\d+(\.\d*)?|\.\d+)$", split[1]) is not None
            optim_params[split[0]] = float(split[1])
    else:
        method = s
        optim_params = {}

    if method == 'adadelta':
        optim_fn = optim.Adadelta
    elif method == 'adagrad':
        optim_fn = optim.Adagrad
    elif method == 'adam':
        optim_fn = optim.Adam
        optim_params['betas'] = (optim_params.get('beta1', 0.5), optim_params.get('beta2', 0.999))
        optim_params.pop('beta1', None)
        optim_params.pop('beta2', None)
    elif method == 'adamax':
        optim_fn = optim.Adamax
    elif method == 'asgd':
        optim_fn = optim.ASGD
    elif method == 'rmsprop':
        optim_fn = optim.RMSprop
    elif method == 'rprop':
        optim_fn = optim.Rprop
    elif method == 'sgd':
        optim_fn = optim.SGD
        assert 'lr' in optim_params
    else:
        raise Exception('Unknown optimization method: "%s"' % method)

    # Valider les paramètres de l'optimiseur
    expected_args = list(inspect.signature(optim_fn.__init__).parameters)
    assert expected_args[:2] == ['self', 'params']
    if not all(k in expected_args[2:] for k in optim_params.keys()):
        raise Exception('Unexpected parameters: expected "%s", got "%s"' % (
            str(expected_args[2:]), str(optim_params.keys())))

    return optim_fn(model.parameters(), **optim_params)
