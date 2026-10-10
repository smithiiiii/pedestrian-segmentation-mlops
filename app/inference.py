import base64
import logging
import time
from pathlib import Path
from typing import Dict, Union

import albumentations as A
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import yaml
from albumentations.pytorch import ToTensorV2

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
DEFAULT_WEIGHTS_PATH = PROJECT_ROOT / "models" / "best_model_b.pt"
DEFAULT_MAX_IMAGE_BYTES = 20 * 1024 * 1024


class InvalidImageError(ValueError):
    """Raised when the supplied bytes are empty, too large, or not a decodable image."""


class SegmentationService:
    """Loads the U-Net once and serves binary pedestrian masks on CPU."""

    def __init__(
        self,
        config_path: Union[str, Path] = DEFAULT_CONFIG_PATH,
        weights_path: Union[str, Path] = DEFAULT_WEIGHTS_PATH,
        max_image_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
    ) -> None:
        config_path, weights_path = Path(config_path), Path(weights_path)
        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / config_path
        if not weights_path.is_absolute():
            weights_path = PROJECT_ROOT / weights_path
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Weights not found: {weights_path}. Train Model B first (python training/train_model_b.py)."
            )

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        model_cfg = config["model"]
        prep_cfg = config["preprocessing"]
        self.image_size = (
            int(prep_cfg["image_size"][0]),
            int(prep_cfg["image_size"][1]),
        )
        self.threshold = float(
            (config.get("inference") or {}).get("default_threshold", 0.5)
        )
        self.max_image_bytes = int(max_image_bytes)
        self.device = torch.device("cpu")

        self.transforms = A.Compose(
            [
                A.Resize(self.image_size[0], self.image_size[1]),
                A.Normalize(mean=tuple(prep_cfg["mean"]), std=tuple(prep_cfg["std"])),
                ToTensorV2(),
            ]
        )

        self.model = smp.Unet(
            encoder_name=model_cfg["encoder_name"],
            encoder_weights=None,
            in_channels=model_cfg["in_channels"],
            classes=model_cfg["classes"],
        )
        try:
            state_dict = torch.load(weights_path, map_location="cpu")
            self.model.load_state_dict(state_dict)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Checkpoint {weights_path.name} does not match the architecture in "
                f"{config_path.name} (encoder={model_cfg['encoder_name']}): {exc}"
            ) from exc
        self.model.to(self.device)
        self.model.eval()

        with torch.no_grad():
            self.model(torch.zeros(1, 3, *self.image_size, device=self.device))

        logger.info(
            "SegmentationService ready (weights=%s, input=%s, threshold=%.2f)",
            weights_path.name,
            self.image_size,
            self.threshold,
        )

    def _decode(self, image_bytes: bytes) -> np.ndarray:
        """Decode raw bytes to an RGB uint8 array; raise InvalidImageError on any failure."""
        if not isinstance(image_bytes, (bytes, bytearray, memoryview)):
            raise InvalidImageError(
                f"Expected bytes, got {type(image_bytes).__name__}."
            )
        if len(image_bytes) == 0:
            raise InvalidImageError("Empty image payload.")
        if len(image_bytes) > self.max_image_bytes:
            raise InvalidImageError(
                f"Image payload too large ({len(image_bytes) / 1e6:.1f} MB; limit {self.max_image_bytes / 1e6:.0f} MB)."
            )

        buffer = np.frombuffer(image_bytes, dtype=np.uint8)
        try:
            image_bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        except cv2.error as exc:
            raise InvalidImageError(f"Could not decode image: {exc}") from exc

        if image_bgr is None or image_bgr.size == 0:
            raise InvalidImageError(
                "Could not decode image: unsupported or corrupted data."
            )
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _encode_png_base64(mask: np.ndarray) -> str:
        success, buffer = cv2.imencode(".png", mask)
        if not success:
            raise RuntimeError("Failed to encode mask as PNG.")
        return base64.b64encode(buffer.tobytes()).decode("ascii")

    def predict(self, image_bytes: bytes) -> Dict[str, object]:
        """Segment pedestrians in an encoded image (JPEG, PNG, BMP, WebP, ...).

        Returns:
            {
              "mask_base64": base64-encoded PNG of the predicted mask at the model's
                             input resolution (config image_size), values 0 / 255,
              "inference_time_ms": milliseconds spent on preprocessing + forward pass
                             + thresholding (excludes decoding and PNG/base64 encoding),
              "original_shape": (height, width) of the decoded input image,
            }

        Raises:
            InvalidImageError: empty, oversized, or undecodable input.
        """
        image = self._decode(image_bytes)
        original_shape = (int(image.shape[0]), int(image.shape[1]))

        start = time.perf_counter()
        tensor = self.transforms(image=image)["image"].unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.sigmoid(self.model(tensor))[0, 0]
        mask = (probs > self.threshold).cpu().numpy().astype(np.uint8) * 255
        inference_time_ms = (time.perf_counter() - start) * 1000.0

        return {
            "mask_base64": self._encode_png_base64(mask),
            "inference_time_ms": round(inference_time_ms, 2),
            "original_shape": original_shape,
        }
