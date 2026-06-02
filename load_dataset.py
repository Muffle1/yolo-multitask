from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset


class ChipsDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transform = transform

        self.image_paths = sorted(
            list(self.images_dir.glob("*.jpg")) +
            list(self.images_dir.glob("*.png")) +
            list(self.images_dir.glob("*.jpeg"))
        )

        self.label_paths = sorted(
            list(self.labels_dir.glob("*.txt"))
        )

        self.labels = None

        if self.label_paths:
            labels = []
            for label_path in self.label_paths:
                with open(label_path, "r", encoding="utf-8") as f:
                    label = f.read().strip()
                    labels.append(label)

            assert len(self.image_paths) == len(self.labels[0]), \
                "Количество фото и label не совпадает"

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.labels is None:
            return image, self.image_paths[idx].name

        label = self.labels[idx]

        return image, label