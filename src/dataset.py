import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2

transform = A.Compose(
    [
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)


def preprocess_image(image_bytes: bytes):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    orig_h, orig_w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    augmented = transform(image=image_rgb)
    tensor_image = augmented["image"].unsqueeze(0)

    return tensor_image, (orig_h, orig_w)


def postprocess_mask(binary_mask: np.ndarray, orig_shape: tuple) -> np.ndarray:
    orig_h, orig_w = orig_shape
    resized_mask = cv2.resize(
        binary_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
    )
    return (resized_mask * 255).astype(np.uint8)
