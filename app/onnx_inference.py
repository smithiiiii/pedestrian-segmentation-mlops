import base64
import logging
import math
import os
import time
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import onnxruntime as ort
import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best_model_b.onnx"
DEFAULT_MAX_IMAGE_BYTES = 20 * 1024 * 1024  # reject absurdly large uploads before decoding


class InvalidImageError(ValueError):
    """Raised when the supplied bytes are empty, too large, or not a decodable image."""


def preprocess_image(
    image_rgb: np.ndarray,
    image_size: Tuple[int, int],
    mean: Sequence[float],
    std: Sequence[float],
) -> np.ndarray:
    """Resize + ImageNet-normalize an RGB uint8 image into a (1, 3, H, W) float32 tensor.

    Matches the training pipeline (Albumentations Resize with bilinear interpolation,
    then Normalize with max_pixel_value=255) without needing Albumentations or PyTorch.
    """
    height, width = image_size
    resized = cv2.resize(image_rgb, (width, height), interpolation=cv2.INTER_LINEAR)
    x = resized.astype(np.float32)
    mean_255 = np.asarray(mean, dtype=np.float32) * 255.0
    inv_std_255 = 1.0 / (np.asarray(std, dtype=np.float32) * 255.0)
    x = (x - mean_255) * inv_std_255  # (H, W, 3), per-channel
    return np.ascontiguousarray(x.transpose(2, 0, 1))[np.newaxis]  # (1, 3, H, W)


class OnnxSegmentationService:
    """Loads the ONNX model once and serves binary pedestrian masks on CPU."""

    def __init__(
        self,
        config_path: Union[str, Path] = DEFAULT_CONFIG_PATH,
        model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
        max_image_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
        intra_op_num_threads: Optional[int] = None,
    ) -> None:
        config_path, model_path = Path(config_path), Path(model_path)
        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / config_path
        if not model_path.is_absolute():
            model_path = PROJECT_ROOT / model_path
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        if not model_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found: {model_path}. Export it first (python training/export_onnx.py)."
            )

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        prep_cfg = config["preprocessing"]
        self.image_size: Tuple[int, int] = (int(prep_cfg["image_size"][0]), int(prep_cfg["image_size"][1]))
        self.mean = tuple(float(v) for v in prep_cfg["mean"])
        self.std = tuple(float(v) for v in prep_cfg["std"])
        self.max_image_bytes = int(max_image_bytes)

        threshold = float((config.get("inference") or {}).get("default_threshold", 0.5))
        if not 0.0 < threshold < 1.0:
            raise ValueError(f"inference.default_threshold must be in (0, 1), got {threshold}")
        self.threshold = threshold
        # sigmoid(z) > t  <=>  z > log(t / (1 - t)); thresholding logits skips a 65k-element sigmoid.
        self.logit_threshold = math.log(threshold / (1.0 - threshold))

        if intra_op_num_threads is None:
            intra_op_num_threads = int(os.environ.get("ORT_INTRA_OP_THREADS", "0"))  # 0 = ORT decides

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.intra_op_num_threads = int(intra_op_num_threads)
        options.inter_op_num_threads = 1

        try:
            self.session = ort.InferenceSession(
                str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
            )
        except Exception as exc:  # noqa: BLE001 - onnxruntime raises several exception types
            raise RuntimeError(f"Failed to load ONNX model {model_path.name}: {exc}") from exc

        model_input = self.session.get_inputs()[0]
        self.input_name: str = model_input.name
        self.output_name: str = self.session.get_outputs()[0].name
        self._check_input_shape(model_input.shape)

        # Warm-up so the first real request doesn't pay one-time allocation/optimization costs.
        dummy = np.zeros((1, 3, *self.image_size), dtype=np.float32)
        for _ in range(2):
            self.session.run([self.output_name], {self.input_name: dummy})

        logger.info(
            "OnnxSegmentationService ready (model=%s, input=%s %s, threshold=%.2f, "
            "intra_op_threads=%s, onnxruntime=%s, providers=%s)",
            model_path.name,
            self.input_name,
            model_input.shape,
            self.threshold,
            intra_op_num_threads or "auto",
            ort.__version__,
            self.session.get_providers(),
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _check_input_shape(self, shape: Sequence[object]) -> None:
        """Fail at startup if the ONNX input does not match the configured image size."""
        if len(shape) != 4:
            raise RuntimeError(f"Expected a 4D (N, C, H, W) ONNX input, got shape {list(shape)}")
        _, channels, height, width = shape
        if isinstance(channels, int) and channels != 3:
            raise RuntimeError(f"ONNX model expects {channels} channels, preprocessing produces 3.")
        for name, onnx_dim, cfg_dim in (("height", height, self.image_size[0]), ("width", width, self.image_size[1])):
            if isinstance(onnx_dim, int) and onnx_dim != cfg_dim:
                raise RuntimeError(
                    f"ONNX model expects {name}={onnx_dim} but configs/config.yaml image_size "
                    f"gives {cfg_dim}. Re-export the model or fix the config."
                )

    def _decode(self, image_bytes: bytes) -> np.ndarray:
        """Decode raw bytes to an RGB uint8 array; raise InvalidImageError on any failure."""
        if not isinstance(image_bytes, (bytes, bytearray, memoryview)):
            raise InvalidImageError(f"Expected bytes, got {type(image_bytes).__name__}.")
        if len(image_bytes) == 0:
            raise InvalidImageError("Empty image payload.")
        if len(image_bytes) > self.max_image_bytes:
            raise InvalidImageError(
                f"Image payload too large ({len(image_bytes) / 1e6:.1f} MB; limit {self.max_image_bytes / 1e6:.0f} MB)."
            )

        buffer = np.frombuffer(image_bytes, dtype=np.uint8)
        try:
            # Default flags honour EXIF orientation, so phone photos come out upright.
            image_bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        except cv2.error as exc:
            raise InvalidImageError(f"Could not decode image: {exc}") from exc

        if image_bgr is None or image_bgr.size == 0:
            raise InvalidImageError("Could not decode image: unsupported or corrupted data.")
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _encode_png_base64(mask: np.ndarray) -> str:
        success, buffer = cv2.imencode(".png", mask)
        if not success:
            raise RuntimeError("Failed to encode mask as PNG.")
        return base64.b64encode(buffer.tobytes()).decode("ascii")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def predict(self, image_bytes: bytes) -> Dict[str, object]:
        """Segment pedestrians in an encoded image (JPEG, PNG, BMP, WebP, ...).

        Returns:
            {
              "mask_base64": base64-encoded PNG of the predicted mask at the model's
                             input resolution (config image_size), values 0 / 255,
              "inference_time_ms": milliseconds spent on preprocessing + ONNX Runtime
                             forward pass + thresholding (excludes decoding and
                             PNG/base64 encoding),
              "original_shape": (height, width) of the decoded input image,
            }

        Raises:
            InvalidImageError: empty, oversized, or undecodable input.
            RuntimeError: ONNX Runtime failed or returned an unexpected output shape.
        """
        image = self._decode(image_bytes)
        original_shape = (int(image.shape[0]), int(image.shape[1]))

        start = time.perf_counter()
        tensor = preprocess_image(image, self.image_size, self.mean, self.std)
        try:
            outputs = self.session.run([self.output_name], {self.input_name: tensor})
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"ONNX Runtime inference failed: {exc}") from exc

        logits = np.squeeze(outputs[0])  # (1, 1, H, W) or (1, H, W) -> (H, W)
        if logits.ndim != 2:
            raise RuntimeError(f"Unexpected model output shape {np.asarray(outputs[0]).shape}.")
        mask = (logits > self.logit_threshold).astype(np.uint8) * 255
        inference_time_ms = (time.perf_counter() - start) * 1000.0

        return {
            "mask_base64": self._encode_png_base64(mask),
            "inference_time_ms": round(inference_time_ms, 2),
            "original_shape": original_shape,
        }
