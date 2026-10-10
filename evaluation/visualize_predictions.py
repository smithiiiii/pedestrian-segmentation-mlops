import sys
from pathlib import Path

import albumentations as A
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import yaml
from albumentations.pytorch import ToTensorV2


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
MANIFEST_PATH = PROJECT_ROOT / "data" / "combined_train_manifest.csv"
WEIGHTS_PATH = PROJECT_ROOT / "models" / "best_model_b.pt"
OUTPUT_PATH = PROJECT_ROOT / "media" / "sample_predictions.png"

NUM_SAMPLES = 5
RANDOM_STATE = 42
THRESHOLD = 0.5
DPI = 300


def resolve(rel_path: str) -> Path:
    """Manifest paths are project-relative; absolute paths are kept as they are."""
    p = Path(rel_path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def read_rgb(path: Path) -> np.ndarray:
    """Read an image as RGB uint8 (ignoring EXIF rotation so it stays aligned with its mask)."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_binary_mask(path: Path) -> np.ndarray:
    """Read a single-channel mask and binarize it (> 127 -> 1) as uint8 {0, 1}."""
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    return (mask > 127).astype(np.uint8)


def main() -> int:

    if not CONFIG_PATH.exists():
        print(f"Config not found: {CONFIG_PATH}", file=sys.stderr)
        return 1
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_cfg = config["model"]
    prep_cfg = config["preprocessing"]
    height, width = int(prep_cfg["image_size"][0]), int(prep_cfg["image_size"][1])
    mean, std = tuple(prep_cfg["mean"]), tuple(prep_cfg["std"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not WEIGHTS_PATH.exists():
        print(
            f"Weights not found: {WEIGHTS_PATH}. Train Model B first (python training/train_model_b.py).",
            file=sys.stderr,
        )
        return 1

    model = smp.Unet(
        encoder_name=model_cfg["encoder_name"],
        encoder_weights=None,
        in_channels=model_cfg["in_channels"],
        classes=model_cfg["classes"],
    )
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.to(device)
    model.eval()

    if not MANIFEST_PATH.exists():
        print(f"Manifest not found: {MANIFEST_PATH}. Run training/prepare_coco_subset.py first.", file=sys.stderr)
        return 1
    df = pd.read_csv(MANIFEST_PATH)
    if df.empty:
        print("Manifest is empty.", file=sys.stderr)
        return 1
    n = min(NUM_SAMPLES, len(df))
    samples = df.sample(n=n, random_state=RANDOM_STATE).reset_index(drop=True)

    infer_tfm = A.Compose(
        [
            A.Resize(height, width),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )

    display_tfm = A.Compose([A.Resize(height, width)])

    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n), squeeze=False)

    for row, record in samples.iterrows():
        image = read_rgb(resolve(record["image_path"]))
        gt_mask = read_binary_mask(resolve(record["mask_path"]))
        if image.shape[:2] != gt_mask.shape[:2]:
            print(f"Image/mask size mismatch for {record['image_path']}", file=sys.stderr)
            return 1

        input_tensor = infer_tfm(image=image)["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.sigmoid(model(input_tensor))[0, 0]
        pred_mask = (probs > THRESHOLD).cpu().numpy().astype(np.uint8)

        shown = display_tfm(image=image, mask=gt_mask)
        shown_image, shown_gt = shown["image"], shown["mask"]

        inter = float((pred_mask & shown_gt).sum())
        union = float((pred_mask | shown_gt).sum())
        iou = (inter + 1e-7) / (union + 1e-7)

        axes[row, 0].imshow(shown_image)
        axes[row, 0].set_title(f"Input ({record['source']}, {record['size_bin']})", fontsize=9)
        axes[row, 1].imshow(shown_gt, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title("Ground Truth Mask", fontsize=9)
        axes[row, 2].imshow(pred_mask, cmap="gray", vmin=0, vmax=1)
        axes[row, 2].set_title(f"Predicted Mask (IoU {iou:.2f})", fontsize=9)

        for col in range(3):
            axes[row, col].axis("off")

    fig.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=DPI)
    plt.close(fig)

    print(f"Device: {device} | samples: {n}")
    print(f"Saved figure to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
