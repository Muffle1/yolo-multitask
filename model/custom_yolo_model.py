import torch
import torch.nn as nn
from ultralytics import YOLO

from .polygon_detect_head import PolygonDetectHead


class CustomYOLOModel(nn.Module):
    def __init__(self, yolo_weights: str, num_classes: int):
        super().__init__()

        self.num_classes = num_classes
        self.yolo = YOLO(yolo_weights).model
        self.saved_features = None

        self.feature_indices = [16, 19, 22]
        self.saved_features = {}

        for idx in self.feature_indices:
            self.yolo.model[idx].register_forward_hook(self._make_hook(idx))

        self.polygon_head = PolygonDetectHead(
            in_channels=[64, 128, 256],
            num_classes=num_classes,
        )

    def _make_hook(self, idx):
        def hook_fn(module, inputs, output):
            self.saved_features[idx] = output
        return hook_fn

    def forward(self, x):
        self.saved_features = {}

        _ = self.yolo(x)

        features = [
            self.saved_features[16],
            self.saved_features[19],
            self.saved_features[22],
        ]

        polygon_output = self.polygon_head(features)

        return polygon_output