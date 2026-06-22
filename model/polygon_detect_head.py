import torch
import torch.nn as nn

class PolygonDetectHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(PolygonDetectHead, self).__init__()
        
        self.num_classes = num_classes
        self.num_points = 4
        self.out_dim = 1 + self.num_classes + self.num_points * 2

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, c, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv2d(c, self.out_dim, kernel_size=1)
            )
            for c in in_channels
        ])

    def forward(self, x):
        return [head(feature) for head, feature in zip(self.heads, x)]