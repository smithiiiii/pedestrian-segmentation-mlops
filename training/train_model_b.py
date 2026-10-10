import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Sequence

os.environ["CV2_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import cv2

cv2.setNumThreads(0)

import albumentations as A
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import yaml
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

torch.set_num_threads(os.cpu_count() or 4)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "combined_train_manifest.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "models" / "best_model_b.pt"
DEFAULT_HISTORY = PROJECT_ROOT / "training" / "history_model_b.json"

TRAINING_DEFAULTS = {
    "batch_size": 16,
    "epochs": 20,
    "learning_rate": 3e-4,
    "num_workers": 0,
    "seed": 42,
}


class CombinedSegmentationDataset(Dataset):
    REQUIRED_COLUMNS = ("image_path", "mask_path", "source", "size_bin")

    def __init__(self, manifest: pd.DataFrame, transform: A.Compose) -> None:
        missing = [c for c in self.REQUIRED_COLUMNS if c not in manifest.columns]
        if missing:
            raise ValueError(f"Manifest is missing required columns: {missing}")
        self.df = manifest.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    @staticmethod
    def _resolve(rel_path: str) -> Path:
        path = Path(rel_path)
        return path if path.is_absolute() else PROJECT_ROOT / path

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        image_path = self._resolve(row["image_path"])
        mask_path = self._resolve(row["mask_path"])

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not read mask: {mask_path}")
        mask = (mask > 127).astype(np.float32)

        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(f"Image/mask size mismatch for {image_path.name}: {image.shape[:2]} vs {mask.shape[:2]}")

        out = self.transform(image=image, mask=mask)
        return out["image"], out["mask"].float().unsqueeze(0)


def get_train_transforms(image_size: Sequence[int], mean: Sequence[float], std: Sequence[float]) -> A.Compose:
    return A.Compose(
        [
            A.Resize(int(image_size[0]), int(image_size[1])),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=tuple(mean), std=tuple(std)),
            ToTensorV2(),
        ]
    )


class BCEDiceLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = smp.losses.DiceLoss(mode="binary", from_logits=True)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce(logits, targets) + self.dice(logits, targets)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def validate_manifest_files(df: pd.DataFrame) -> None:
    missing = []
    for col in ("image_path", "mask_path"):
        for rel in df[col]:
            if not CombinedSegmentationDataset._resolve(rel).exists():
                missing.append(rel)
    if missing:
        preview = "\n  ".join(missing[:5])
        raise FileNotFoundError(
            f"{len(missing)} files listed in the manifest do not exist. First few:\n  {preview}\n"
            "Re-run training/prepare_coco_subset.py."
        )


def save_history(path: Path, history: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Model B on the combined manifest.")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    print("[STEP 1/5] Loading configuration...", flush=True)
    config_path = resolve(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        return 1
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_cfg = config["model"]
    prep_cfg = config["preprocessing"]
    train_cfg: Dict[str, float] = dict(TRAINING_DEFAULTS)
    train_cfg.update(config.get("training") or {})

    epochs = int(args.epochs if args.epochs is not None else train_cfg["epochs"])
    batch_size = int(args.batch_size if args.batch_size is not None else train_cfg["batch_size"])
    lr = float(train_cfg["learning_rate"])
    seed = int(train_cfg["seed"])

    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[STEP 2/5] Preparing Dataset...", flush=True)
    manifest_path = resolve(args.manifest)
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    df = pd.read_csv(manifest_path)
    validate_manifest_files(df)
    dataset = CombinedSegmentationDataset(
        df, get_train_transforms(prep_cfg["image_size"], prep_cfg["mean"], prep_cfg["std"])
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        drop_last=len(dataset) > batch_size,
    )

    print("[STEP 3/5] Initializing U-Net Architecture...", flush=True)
    model = smp.Unet(
        encoder_name=model_cfg["encoder_name"],
        encoder_weights="imagenet",
        in_channels=model_cfg["in_channels"],
        classes=model_cfg["classes"],
    ).to(device)

    print("[STEP 4/5] Setting up Optimizer & Loss...", flush=True)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history: Dict[str, object] = {
        "metadata": {
            "device": str(device),
            "num_samples": len(dataset),
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": lr,
        },
        "epoch": [],
        "train_loss": [],
        "learning_rate": [],
    }

    start = time.time()
    total_batches = len(loader)

    print(f"\n[STEP 5/5] READY! Starting Epoch 1/{epochs} ({total_batches} batches total)...", flush=True)

    for epoch in range(1, epochs + 1):
        model.train()
        running, seen = 0.0, 0
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"\n=== Epoch {epoch}/{epochs} ===", flush=True)
        for batch_idx, (images, masks) in enumerate(loader, 1):
            batch_start = time.time()
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            running += loss.item() * images.size(0)
            seen += images.size(0)

            batch_time = time.time() - batch_start
            print(
                f"  Batch [{batch_idx:02d}/{total_batches:02d}] Finished in {batch_time:.2f}s | Loss: {loss.item():.4f}",
                flush=True,
            )

        scheduler.step()
        epoch_loss = running / max(seen, 1)

        history["epoch"].append(epoch)
        history["train_loss"].append(epoch_loss)
        history["learning_rate"].append(current_lr)
        save_history(resolve(args.history), history)

        print(
            f"=== Epoch {epoch:02d} Complete | Avg Loss: {epoch_loss:.4f} | Total Time: {time.time() - start:.0f}s ===",
            flush=True,
        )

    output_path = resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)

    print(f"\nSaved final weights to {output_path}", flush=True)
    print(f"Saved history to {resolve(args.history)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
