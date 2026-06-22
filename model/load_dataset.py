from pathlib import Path
import yaml
import cv2
import torch
from torch.utils.data import Dataset


class PolygonDataset(Dataset):
    def __init__(self, data_yaml: str, split: str = "train", imgsz: int = 640):
        self.imgsz = imgsz

        with open(data_yaml, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        root = Path(cfg["path"])
        image_dir = root / cfg[split]
        label_dir = root / "labels" / split

        self.images = sorted(
            list(image_dir.glob("*.jpg")) +
            list(image_dir.glob("*.png")) +
            list(image_dir.glob("*.jpeg"))
        )

        self.label_dir = label_dir
        self.names = cfg["names"]
        self.num_classes = len(self.names)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label_path = self.label_dir / f"{img_path.stem}.txt"

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.imgsz, self.imgsz))

        img = torch.from_numpy(img).float() / 255.0
        img = img.permute(2, 0, 1)

        labels = []

        if label_path.exists():
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    values = list(map(float, line.strip().split()))
                    if len(values) != 9:
                        raise ValueError(f"Bad label line in {label_path}: {line}")
                    labels.append(values)

        labels = torch.tensor(labels, dtype=torch.float32)

        return img, labels


def polygon_collate_fn(batch):
    images, labels = zip(*batch)
    return torch.stack(images, dim=0), list(labels)