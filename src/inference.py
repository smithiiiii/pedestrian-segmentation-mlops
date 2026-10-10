from pathlib import Path
from typing import Optional

import numpy as np
import segmentation_models_pytorch as smp
import torch
import yaml
from PIL import Image

from src.preprocessing import get_inference_transforms, load_image_from_bytes

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


class InferenceEngine:
    """Loads config + weights once and serves predictions.

    Usage:
        engine = InferenceEngine()
        mask = engine.predict(image_bytes, threshold=0.5)
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        cfg_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        if not cfg_path.is_absolute():
            cfg_path = PROJECT_ROOT / cfg_path
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")

        with open(cfg_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        model_cfg = self.config["model"]
        prep_cfg = self.config["preprocessing"]
        self.default_threshold = float(self.config["inference"]["default_threshold"])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.transforms = get_inference_transforms(
            prep_cfg["image_size"], prep_cfg["mean"], prep_cfg["std"]
        )

        self.model = smp.Unet(
            encoder_name=model_cfg["encoder_name"],
            encoder_weights=model_cfg["encoder_weights"],
            in_channels=model_cfg["in_channels"],
            classes=model_cfg["classes"],
        )

        ckpt_path = Path(model_cfg["checkpoint_path"])
        if not ckpt_path.is_absolute():
            ckpt_path = PROJECT_ROOT / ckpt_path
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}. Place the file downloaded from Colab at models/best_model.pt."
            )

        state_dict = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, image_bytes: bytes, threshold: float = None) -> np.ndarray:
        """Segment pedestrians in an encoded image.

        Args:
            image_bytes: raw encoded image bytes.
            threshold: probability cutoff in [0, 1]; defaults to the config value.

        Returns:
            2D uint8 array of shape (orig_h, orig_w) with values 0 (background)
            and 255 (pedestrian), ready for ``cv2.imencode('.png', mask)``.

        Raises:
            ValueError: for undecodable images or an out-of-range threshold.
        """
        if threshold is None:
            threshold = self.default_threshold
        threshold = float(threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")

        image, (orig_w, orig_h) = load_image_from_bytes(image_bytes)
        tensor = self.transforms(image=image)["image"].unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.sigmoid(logits)[0, 0].cpu().numpy()

        mask = (probs > threshold).astype(np.uint8) * 255
        mask_img = Image.fromarray(mask, mode="L").resize(
            (orig_w, orig_h), Image.NEAREST
        )
        return np.array(mask_img, dtype=np.uint8)
