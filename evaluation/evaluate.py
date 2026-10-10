import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import albumentations as A
import numpy as np
import segmentation_models_pytorch as smp
import torch
import yaml
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "PennFudanPed"
DEFAULT_OUTPUT = PROJECT_ROOT / "evaluation" / "eval_results.json"
DEFAULT_THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70]
EPS = 1e-7
METRIC_NAMES = ("iou", "dice", "precision", "recall")


class PennFudanEvalDataset(Dataset):
    """Penn-Fudan images + binary masks with deterministic preprocessing."""

    def __init__(
        self,
        root: Path,
        files: Sequence[str],
        image_size: Sequence[int],
        mean: Sequence[float],
        std: Sequence[float],
    ) -> None:
        self.img_dir = root / "PNGImages"
        self.mask_dir = root / "PedMasks"
        self.files = list(files)
        self.transform = A.Compose(
            [
                A.Resize(int(image_size[0]), int(image_size[1])),
                A.Normalize(mean=tuple(mean), std=tuple(std)),
                ToTensorV2(),
            ]
        )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        name = self.files[idx]
        image = np.array(Image.open(self.img_dir / name).convert("RGB"), dtype=np.uint8)
        mask_path = self.mask_dir / name.replace(".png", "_mask.png")
        mask = np.array(Image.open(mask_path))
        mask = (mask > 0).astype(np.uint8)

        out = self.transform(image=image, mask=mask)
        return out["image"], out["mask"].float(), name


def list_split_files(root: Path, split: str, val_split: float, seed: int) -> List[str]:
    """Reproduce the training script's split."""
    img_dir = root / "PNGImages"
    if not img_dir.is_dir():
        raise FileNotFoundError(
            f"Image directory not found: {img_dir}. Use --data-root to point at the unzipped PennFudanPed folder."
        )
    files = sorted(f.name for f in img_dir.iterdir() if f.suffix.lower() == ".png")
    if not files:
        raise FileNotFoundError(f"No .png images found in {img_dir}")

    if split == "all":
        return files

    rng = random.Random(seed)
    shuffled = files[:]
    rng.shuffle(shuffled)
    n_val = int(len(shuffled) * val_split)
    return shuffled[:n_val] if split == "val" else shuffled[n_val:]


def load_model(config: dict, device: torch.device) -> torch.nn.Module:
    model_cfg = config["model"]
    model = smp.Unet(
        encoder_name=model_cfg["encoder_name"],
        encoder_weights=model_cfg["encoder_weights"],
        in_channels=model_cfg["in_channels"],
        classes=model_cfg["classes"],
    )
    ckpt = Path(model_cfg["checkpoint_path"])
    if not ckpt.is_absolute():
        ckpt = PROJECT_ROOT / ckpt
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device)
    model.eval()
    return model


def per_image_metrics(probs: torch.Tensor, target: torch.Tensor, threshold: float) -> Dict[str, torch.Tensor]:
    """Compute IoU, Dice, Precision, Recall for each image in a batch.

    Args:
        probs: (B, H, W) probabilities in [0, 1].
        target: (B, H, W) binary ground truth {0., 1.}.
        threshold: pixels with prob > threshold are predicted foreground.

    Epsilon is added to both numerator and denominator, so if an image has no
    positive pixels in both prediction and ground truth, every metric is 1.0;
    if only one of them is empty, the relevant metrics are ~0 (no division by zero).
    """
    pred = (probs > threshold).float()
    dims = (1, 2)
    tp = (pred * target).sum(dim=dims)
    fp = (pred * (1.0 - target)).sum(dim=dims)
    fn = ((1.0 - pred) * target).sum(dim=dims)

    return {
        "iou": (tp + EPS) / (tp + fp + fn + EPS),
        "dice": (2.0 * tp + EPS) / (2.0 * tp + fp + fn + EPS),
        "precision": (tp + EPS) / (tp + fp + EPS),
        "recall": (tp + EPS) / (tp + fn + EPS),
    }


@torch.no_grad()
def run_evaluation(
    model: torch.nn.Module,
    loader: DataLoader,
    thresholds: Sequence[float],
    device: torch.device,
) -> Dict[float, Dict[str, np.ndarray]]:
    """Single forward pass per batch; metrics computed for every threshold."""
    collected: Dict[float, Dict[str, List[np.ndarray]]] = {t: {m: [] for m in METRIC_NAMES} for t in thresholds}

    for images, masks, _names in loader:
        images = images.to(device)
        masks = masks.to(device)
        probs = torch.sigmoid(model(images))[:, 0]

        for t in thresholds:
            scores = per_image_metrics(probs, masks, t)
            for m in METRIC_NAMES:
                collected[t][m].append(scores[m].cpu().numpy())

    return {t: {m: np.concatenate(collected[t][m]) for m in METRIC_NAMES} for t in thresholds}


def summarize(per_image: Dict[float, Dict[str, np.ndarray]]) -> List[dict]:
    rows = []
    for t, metrics in per_image.items():
        row = {"threshold": float(t)}
        for m, values in metrics.items():
            ddof = 1 if len(values) > 1 else 0
            row[f"mean_{m}"] = float(np.mean(values))
            row[f"std_{m}"] = float(np.std(values, ddof=ddof))
        rows.append(row)
    return rows


def print_table(rows: List[dict], n_samples: int) -> None:
    header = f"{'Threshold':>9} | {'Mean IoU':>17} | {'Mean Dice':>17} | {'Precision':>17} | {'Recall':>17}"
    line = "-" * len(header)
    print(f"\nResults over {n_samples} images (mean ± std)\n")
    print(header)
    print(line)
    for r in rows:
        cells = [f"{r[f'mean_{m}']:.4f} ± {r[f'std_{m}']:.4f}" for m in METRIC_NAMES]
        print(f"{r['threshold']:>9.2f} | {cells[0]:>17} | {cells[1]:>17} | {cells[2]:>17} | {cells[3]:>17}")
    print(line)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate the pedestrian U-Net with a threshold sweep.")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument(
        "--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Path to the unzipped PennFudanPed folder."
    )
    p.add_argument("--split", choices=["val", "train", "all"], default="val")
    p.add_argument("--val-split", type=float, default=0.2, help="Must match training.")
    p.add_argument("--seed", type=int, default=42, help="Must match training.")
    p.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0, help="0 is safest on Windows.")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    for t in args.thresholds:
        if not 0.0 <= t <= 1.0:
            print(f"Invalid threshold {t}: must be within [0, 1].", file=sys.stderr)
            return 1

    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        return 1
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    prep = config["preprocessing"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = args.data_root if args.data_root.is_absolute() else PROJECT_ROOT / args.data_root

    files = list_split_files(data_root, args.split, args.val_split, args.seed)
    dataset = PennFudanEvalDataset(data_root, files, prep["image_size"], prep["mean"], prep["std"])
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = load_model(config, device)
    print(f"Device: {device} | split: {args.split} | images: {len(dataset)} | input size: {tuple(prep['image_size'])}")
    if args.split == "all":
        print("WARNING: 'all' includes training images; scores will be optimistic.")

    start = time.time()
    per_image = run_evaluation(model, loader, args.thresholds, device)
    elapsed = time.time() - start

    rows = summarize(per_image)
    print_table(rows, len(dataset))

    best = max(rows, key=lambda r: r["mean_iou"])
    print(
        f"\nOptimal threshold (max Mean IoU): {best['threshold']:.2f} "
        f"-> Mean IoU = {best['mean_iou']:.4f}, Mean Dice = {best['mean_dice']:.4f}"
    )
    print(f"Evaluation time: {elapsed:.1f}s")

    output_path = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint": str(config["model"]["checkpoint_path"]),
            "device": str(device),
            "split": args.split,
            "num_samples": len(dataset),
            "image_size": list(prep["image_size"]),
            "split_seed": args.seed,
            "val_split": args.val_split,
            "epsilon": EPS,
            "std_type": "sample (ddof=1)",
        },
        "thresholds": rows,
        "optimal": {
            "criterion": "max mean_iou",
            "threshold": best["threshold"],
            "mean_iou": best["mean_iou"],
            "mean_dice": best["mean_dice"],
            "mean_precision": best["mean_precision"],
            "mean_recall": best["mean_recall"],
        },
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved results to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
