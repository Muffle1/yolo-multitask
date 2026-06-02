from torch.utils.data import DataLoader, Subset
from torchvision import transforms, models

from ultralytics import YOLO

def main():
    yolo = YOLO("yolov8n.pt")
    core = yolo.model
    for i, layer in enumerate(core.model):
        print(i, layer.__class__.__name__)


if __name__ == "__main__":
    main()