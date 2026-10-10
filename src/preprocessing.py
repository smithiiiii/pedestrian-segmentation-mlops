import io
from typing import Sequence, Tuple

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from PIL import Image, UnidentifiedImageError


def get_inference_transforms(
    image_size: Sequence[int],
    mean: Sequence[float],
    std: Sequence[float],
) -> A.Compose:
    """Build the deterministic inference transform.

    Args:
        image_size: (height, width) the model was trained on.
        mean: per-channel normalization mean (RGB order).
        std: per-channel normalization std (RGB order).

    Returns:
        Albumentations pipeline: call as ``tfm(image=rgb_uint8_array)["image"]``
        to get a float tensor of shape (3, H, W).
    """
    height, width = int(image_size[0]), int(image_size[1])
    return A.Compose(
        [
            A.Resize(height, width),
            A.Normalize(mean=tuple(mean), std=tuple(std)),
            ToTensorV2(),
        ]
    )


def load_image_from_bytes(image_bytes: bytes) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Decode raw image bytes into an RGB uint8 array.

    Args:
        image_bytes: raw bytes of an encoded image (PNG, JPEG, ...).

    Returns:
        (image, (orig_w, orig_h)) where ``image`` has shape (H, W, 3), dtype uint8.

    Raises:
        ValueError: if the bytes are empty or are not a decodable image.
    """
    if not image_bytes:
        raise ValueError("Empty image payload.")
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img = img.convert("RGB")
            image = np.array(img, dtype=np.uint8)
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Could not decode image: {exc}") from exc

    orig_h, orig_w = image.shape[:2]
    return image, (orig_w, orig_h)
