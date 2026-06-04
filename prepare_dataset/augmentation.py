import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import AugmentationConfig
from .parser import parse_label_file, parse_label_line


class Augmentation:
    def __init__(self, config: AugmentationConfig, output_dir: Path):
        self.config = config

        output_images = output_dir / "images"
        output_labels = output_dir / "labels"

        self.output_images_train = output_images / "train"
        self.output_labels_train = output_labels / "train"
        self.output_images_val = output_images / "val"
        self.output_labels_val = output_labels / "val"

        for p in [
            self.output_images_train,
            self.output_labels_train,
            self.output_images_val,
            self.output_labels_val,
        ]:
            p.mkdir(parents=True, exist_ok=True)

        self.output_images_with_boxes = output_dir / "images_with_boxes"
        self.output_images_with_boxes.mkdir(parents=True, exist_ok=True)

    def build_affine_matrix(
        self,
        width: int,
        height: int,
        config: AugmentationConfig,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        angle = random.uniform(*config.rotation_range) if config.rotation_range else 0.0
        scale = 1.0 + random.uniform(-config.zoom_range, config.zoom_range)

        hflip = bool(config.horizontal_flip and random.choice([True, False]))
        vflip = bool(config.vertical_flip and random.choice([True, False]))

        cx = (width - 1) / 2.0
        cy = (height - 1) / 2.0

        matrix = np.eye(3, dtype=np.float32)

        if hflip:
            hflip_matrix = np.array(
                [
                    [-1, 0, width - 1],
                    [0, 1, 0],
                    [0, 0, 1],
                ],
                dtype=np.float32,
            )
            matrix = hflip_matrix @ matrix

        if vflip:
            vflip_matrix = np.array(
                [
                    [1, 0, 0],
                    [0, -1, height - 1],
                    [0, 0, 1],
                ],
                dtype=np.float32,
            )
            matrix = vflip_matrix @ matrix

        rotate_scale = cv2.getRotationMatrix2D((cx, cy), angle, scale)
        rotate_scale_3x3 = np.eye(3, dtype=np.float32)
        rotate_scale_3x3[:2] = rotate_scale

        matrix = rotate_scale_3x3 @ matrix

        ops = {
            "angle": angle,
            "scale": scale,
            "hflip": hflip,
            "vflip": vflip,
        }

        return matrix[:2], ops

    def apply_geometric_transforms(
        self,
        image: np.ndarray,
        config: AugmentationConfig,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        height, width = image.shape[:2]

        affine_matrix, ops = self.build_affine_matrix(width, height, config)

        transformed = cv2.warpAffine(
            image,
            affine_matrix,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        return transformed, affine_matrix, ops

    def apply_visual_transforms(
        self,
        image: np.ndarray,
        config: AugmentationConfig,
    ) -> np.ndarray:
        img = image.astype(np.float32)

        brightness = random.uniform(*config.brightness_range)
        contrast = random.uniform(*config.contrast_range)
        saturation = random.uniform(*config.saturation_range)
        hue = random.uniform(*config.hue_range)

        img = img * brightness
        img = (img - 127.5) * contrast + 127.5
        img = np.clip(img, 0, 255).astype(np.uint8)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

        hsv[..., 1] *= saturation

        hsv[..., 0] = (hsv[..., 0] + hue) % 180

        hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2], 0, 255)

        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return img
    
    def apply_noise_and_blur(self, image: np.ndarray, config) -> np.ndarray:
        img = image.copy()

        if getattr(config, "noise_std_range", None):
            std = random.uniform(*config.noise_std_range)

            noise = np.random.normal(
                loc=0,
                scale=std,
                size=img.shape,
            ).astype(np.float32)

            img = img.astype(np.float32) + noise
            img = np.clip(img, 0, 255).astype(np.uint8)

        if getattr(config, "blur_kernel_range", None):
            kernel = random.choice(config.blur_kernel_range)

            if kernel > 1:
                if kernel % 2 == 0:
                    kernel += 1

                img = cv2.GaussianBlur(
                    img,
                    ksize=(kernel, kernel),
                    sigmaX=0,
                )

        return img

    def transform_coordinates(
        self,
        entry: dict[str, Any],
        image_size: tuple[int, int],
        affine_matrix: np.ndarray,
    ) -> dict[str, Any]:
        width, height = image_size
        result = dict(entry)

        points = []

        for idx in range(1, 5):
            x_key = f"x{idx}"
            y_key = f"y{idx}"

            if x_key in entry and y_key in entry:
                x = float(entry[x_key]) * width
                y = float(entry[y_key]) * height
                points.append((idx, x, y))

        if not points:
            return result

        matrix_3x3 = np.eye(3, dtype=np.float32)
        matrix_3x3[:2] = affine_matrix

        for idx, x, y in points:
            point = np.array([x, y, 1.0], dtype=np.float32)
            x_new, y_new, _ = matrix_3x3 @ point

            x_new = float(np.clip(x_new, 0, width - 1))
            y_new = float(np.clip(y_new, 0, height - 1))

            result[f"x{idx}"] = round(x_new / width, 6)
            result[f"y{idx}"] = round(y_new / height, 6)

        return result

    def serialize_label(self, entry: dict[str, Any], label_format: list[str]) -> str:
        values = []

        for name in label_format:
            value = entry.get(name, "")

            if isinstance(value, float) and value.is_integer():
                values.append(str(int(value)))
            else:
                values.append(str(value))

        return " ".join(values)

    def save_label_file(self, label_path: Path, lines: list[str]) -> None:
        label_path.parent.mkdir(parents=True, exist_ok=True)

        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))

    def draw_bounding_boxes(
        self,
        image: np.ndarray,
        labels: list[str],
        suffix: str,
    ) -> None:
        image_cp = image.copy()
        height, width = image_cp.shape[:2]

        for line in labels:
            parsed = parse_label_line(line, self.config.label_format)

            if all(k in parsed for k in ("x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4")):
                box = []

                for idx in range(1, 5):
                    x = int(float(parsed[f"x{idx}"]) * width)
                    y = int(float(parsed[f"y{idx}"]) * height)
                    box.append((x, y))

                if (parsed["class"] == "0"):
                    cv2.polylines(
                        image_cp,
                        [np.array(box, dtype=np.int32)],
                        isClosed=True,
                        color=(0, 0, 255),
                        thickness=2,
                    )
                elif (parsed["class"] == "1"):
                    cv2.polylines(
                        image_cp,
                        [np.array(box, dtype=np.int32)],
                        isClosed=True,
                        color=(0, 255, 0),
                        thickness=2,
                    )
                else:
                    cv2.polylines(
                        image_cp,
                        [np.array(box, dtype=np.int32)],
                        isClosed=True,
                        color=(255, 0, 0),
                        thickness=2,
                    )

        cv2.imwrite(
            str(self.output_images_with_boxes / f"image{suffix}.png"),
            image_cp,
        )

    def augment(self, images_dir, labels_dir):
        image_paths = list(images_dir)

        i = 0

        for image_path in image_paths:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

            if image is None:
                print(f"Skip unreadable image: {image_path}")
                continue

            label_path = labels_dir / f"{image_path.stem}.txt"
            labels = parse_label_file(label_path) if label_path.exists() else []

            for _ in range(self.config.amount):
                augmented_image, affine_matrix, ops = self.apply_geometric_transforms(
                    image,
                    self.config,
                )

                augmented_image = self.apply_visual_transforms(
                    augmented_image,
                    self.config,
                )

                augmented_image = self.apply_noise_and_blur(
                    augmented_image,
                    self.config,
                )

                suffix = f"_aug{i + 1}"
                target_image_name = f"image{suffix}.png"

                is_train = random.random() < self.config.train_val_split

                if is_train:
                    image_out_path = self.output_images_train / target_image_name
                    label_out_path = self.output_labels_train / f"image{suffix}.txt"
                else:
                    image_out_path = self.output_images_val / target_image_name
                    label_out_path = self.output_labels_val / f"image{suffix}.txt"

                cv2.imwrite(str(image_out_path), augmented_image)

                transformed_labels = []

                if labels and self.config.label_format:
                    for line in labels:
                        parsed = parse_label_line(line, self.config.label_format)

                        if "raw" in parsed:
                            transformed_labels.append(parsed["raw"])
                        else:
                            transformed = self.transform_coordinates(
                                parsed,
                                (augmented_image.shape[1], augmented_image.shape[0]),
                                affine_matrix,
                            )
                            transformed_labels.append(
                                self.serialize_label(
                                    transformed,
                                    self.config.label_format,
                                )
                            )

                    self.save_label_file(label_out_path, transformed_labels)
                    self.draw_bounding_boxes(
                        augmented_image,
                        transformed_labels,
                        suffix,
                    )

                i += 1