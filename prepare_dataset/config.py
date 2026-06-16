from dataclasses import dataclass

@dataclass
class AugmentationConfig:
    """Configuration for data augmentation."""
    horizontal_flip: bool = True
    vertical_flip: bool = False
    rotation_range: tuple[int, int] = (-15, 15)
    zoom_range: float = 0.1
    brightness_range: tuple[float, float] = (0.8, 1.2)
    contrast_range: tuple[float, float] = (0.8, 1.2)
    saturation_range: tuple[float, float] = (0.8, 1.2)
    hue_range: tuple[float, float] = (0.8, 1.2)
    noise_std_range: tuple[float, float] = (0.0, 6.0)
    blur_kernel_range: list[int] = (1, 1, 3, 3, 5)
    glare_chance: float = 0.3
    amount: int = 100
    label_format: list[str] | None = None
    train_val_split: float = 0.8
