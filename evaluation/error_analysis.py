import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
 
import albumentations as A
import matplotlib
 
matplotlib.use("Agg")  
import matplotlib.pyplot as plt  
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
DEFAULT_ARTIFACTS = PROJECT_ROOT / "evaluation" / "artifacts"
DEFAULT_REPORT = PROJECT_ROOT / "evaluation" / "error_report.json"
EPS = 1e-7
 
SMALL = "Small Pedestrians"
MEDIUM = "Medium/Full-Body Pedestrians"
LARGE = "Large/Close-Up Pedestrians"
CATEGORY_ORDER = (SMALL, MEDIUM, LARGE)
 
 



class PennFudanEvalDataset(Dataset):
    """Penn-Fudan images + binary masks with deterministic preprocessing."""
 
    def __init__(self, root: Path, files: Sequence[str], image_size: Sequence[int],
                 mean: Sequence[float], std: Sequence[float]) -> None:
        self.img_dir = root / "PNGImages"
        self.mask_dir = root / "PedMasks"
        self.files = list(files)
        self.transform = A.Compose([
            A.Resize(int(image_size[0]), int(image_size[1])),
            A.Normalize(mean=tuple(mean), std=tuple(std)),
            ToTensorV2(),
        ])
 
    def __len__(self) -> int:
        return len(self.files)
 
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        name = self.files[idx]
        image = np.array(Image.open(self.img_dir / name).convert("RGB"), dtype=np.uint8)
        mask = np.array(Image.open(self.mask_dir / name.replace(".png", "_mask.png")))
        mask = (mask > 0).astype(np.uint8)  
        out = self.transform(image=image, mask=mask)
        return out["image"], out["mask"].float()
 
 
def list_split_files(root: Path, split: str, val_split: float, seed: int) -> List[str]:
    """Reproduce the split used by the Colab training script."""
    img_dir = root / "PNGImages"
    if not img_dir.is_dir():
        raise FileNotFoundError(
            f"Image directory not found: {img_dir}. Use --data-root to point at the "
            "unzipped PennFudanPed folder."
        )
    files = sorted(f.name for f in img_dir.iterdir() if f.suffix.lower() == ".png")
    if not files:
        raise FileNotFoundError(f"No .png images found in {img_dir}")
    if split == "all":
        return files
    shuffled = files[:]
    random.Random(seed).shuffle(shuffled)
    n_val = int(len(shuffled) * val_split)
    return shuffled[:n_val] if split == "val" else shuffled[n_val:]
 
 
def load_model(config: dict, device: torch.device) -> torch.nn.Module:
    cfg = config["model"]
    model = smp.Unet(
        encoder_name=cfg["encoder_name"],
        encoder_weights=cfg["encoder_weights"],
        in_channels=cfg["in_channels"],
        classes=cfg["classes"],
    )
    ckpt = Path(cfg["checkpoint_path"])
    if not ckpt.is_absolute():
        ckpt = PROJECT_ROOT / ckpt
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device)
    model.eval()
    return model
 
 



def categorize(gt_area_frac: float) -> str:
    if gt_area_frac < 0.15:
        return SMALL
    if gt_area_frac <= 0.50:
        return MEDIUM
    return LARGE
 
 
def counts(pred: torch.Tensor, gt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-image TP, FP, FN for (B, H, W) binary tensors."""
    dims = (1, 2)
    tp = (pred * gt).sum(dim=dims)
    fp = (pred * (1.0 - gt)).sum(dim=dims)
    fn = ((1.0 - pred) * gt).sum(dim=dims)
    return tp, fp, fn
 
 
def diagnose(gt_area: float, pred_area: float, iou: float) -> str:
    """Coarse, rule-based failure label to speed up manual review."""
    if iou >= 0.7:
        return "good"
    if pred_area < 0.5 * gt_area:
        return "under-segmentation (missed pedestrian pixels)"
    if pred_area > 2.0 * gt_area:
        return "over-segmentation (false positive regions)"
    return "boundary/overlap error (similar area, poor alignment)"
 
 
@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    threshold: float,
    device: torch.device,
) -> Tuple[List[dict], List[np.ndarray]]:
    """Plain sigmoid thresholding. Returns per-image records and binary predictions."""
    records: List[dict] = []
    pred_masks: List[np.ndarray] = []
 
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        probs = torch.sigmoid(model(images))[:, 0]
        pred = (probs > threshold).float()  
 
        tp, fp, fn = counts(pred, masks)
        iou = (tp + EPS) / (tp + fp + fn + EPS)
        dice = (2 * tp + EPS) / (2 * tp + fp + fn + EPS)
        precision = (tp + EPS) / (tp + fp + EPS)
        recall = (tp + EPS) / (tp + fn + EPS)
 
        
        tp_i, fp_i, fn_i = counts(1.0 - pred, masks)
        iou_inv = (tp_i + EPS) / (tp_i + fp_i + fn_i + EPS)
 
        n_pix = float(masks.shape[1] * masks.shape[2])
        gt_area = (masks.sum(dim=(1, 2)) / n_pix).cpu().numpy()
        pred_area = (pred.sum(dim=(1, 2)) / n_pix).cpu().numpy()
        max_prob = probs.amax(dim=(1, 2)).cpu().numpy()
 
        iou, dice = iou.cpu().numpy(), dice.cpu().numpy()
        precision, recall = precision.cpu().numpy(), recall.cpu().numpy()
        iou_inv = iou_inv.cpu().numpy()
 
        for i in range(images.shape[0]):
            records.append({
                "iou": float(iou[i]),
                "dice": float(dice[i]),
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "gt_area_frac": float(gt_area[i]),
                "pred_area_frac": float(pred_area[i]),
                "max_prob": float(max_prob[i]),
                "iou_if_inverted": float(iou_inv[i]),
                "category": categorize(float(gt_area[i])),
            })
            pred_masks.append(pred[i].cpu().numpy().astype(np.uint8))
 
    return records, pred_masks
 
 
def category_stats(records: List[dict]) -> Dict[str, dict]:
    stats: Dict[str, dict] = {}
    for name in CATEGORY_ORDER:
        subset = [r for r in records if r["category"] == name]
        if subset:
            stats[name] = {
                "count": len(subset),
                "mean_iou": float(np.mean([r["iou"] for r in subset])),
                "mean_dice": float(np.mean([r["dice"] for r in subset])),
            }
        else:
            stats[name] = {"count": 0, "mean_iou": None, "mean_dice": None}
    return stats
 
 



def denormalize(tensor: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> np.ndarray:
    m = torch.tensor(mean).view(3, 1, 1)
    s = torch.tensor(std).view(3, 1, 1)
    return (tensor.cpu() * s + m).clamp(0, 1).permute(1, 2, 0).numpy()
 
 
def save_comparison(
    path: Path,
    image: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    record: dict,
    name: str,
    tag: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4))
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[1].imshow(gt, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"Ground Truth Mask ({record['gt_area_frac'] * 100:.1f}% of frame)")
    axes[2].imshow(pred, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(f"Predicted Mask ({record['pred_area_frac'] * 100:.1f}% of frame)")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(
        f"{tag}: {name} | {record['category']} | IoU {record['iou']:.3f} | "
        f"Dice {record['dice']:.3f} | P {record['precision']:.3f} | R {record['recall']:.3f}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
 
 



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Error analysis for the pedestrian U-Net.")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--split", choices=["val", "train", "all"], default="val")
    p.add_argument("--val-split", type=float, default=0.2, help="Must match training.")
    p.add_argument("--seed", type=int, default=42, help="Must match training.")
    p.add_argument("--threshold", type=float, default=None,
                   help="Defaults to inference.default_threshold from the config (0.50).")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0, help="0 is safest on Windows.")
    p.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS)
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return p.parse_args()
 
 
def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path
 
 
def case_entry(rank: int, idx: int, files: Sequence[str], records: List[dict],
               artifact: Path) -> dict:
    r = records[idx]
    return {
        "rank": rank,
        "image": files[idx],
        "category": r["category"],
        "iou": r["iou"],
        "dice": r["dice"],
        "precision": r["precision"],
        "recall": r["recall"],
        "gt_area_pct": r["gt_area_frac"] * 100.0,
        "pred_area_pct": r["pred_area_frac"] * 100.0,
        "max_probability": r["max_prob"],
        "diagnosis": diagnose(r["gt_area_frac"], r["pred_area_frac"], r["iou"]),
        "artifact": str(artifact.relative_to(PROJECT_ROOT)) if artifact.is_relative_to(PROJECT_ROOT)
        else str(artifact),
    }
 
 
def main() -> int:
    args = parse_args()
 
    config_path = resolve(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        return 1
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
 
    prep = config["preprocessing"]
    threshold: float = (
        args.threshold if args.threshold is not None
        else float(config["inference"]["default_threshold"])
    )
    if not 0.0 <= threshold <= 1.0:
        print(f"Invalid threshold {threshold}: must be within [0, 1].", file=sys.stderr)
        return 1
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = resolve(args.data_root)
    files = list_split_files(data_root, args.split, args.val_split, args.seed)
    dataset = PennFudanEvalDataset(data_root, files, prep["image_size"], prep["mean"], prep["std"])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
 
    model = load_model(config, device)
    print(f"Device: {device} | split: {args.split} | images: {len(dataset)} | "
          f"threshold: P > {threshold:.2f}")
    if args.split == "all":
        print("WARNING: 'all' includes training images; scores will be optimistic.")
 
    records, pred_masks = run_inference(model, loader, threshold, device)
 
    
    stats = category_stats(records)
    overall = {
        "count": len(records),
        "mean_iou": float(np.mean([r["iou"] for r in records])),
        "mean_dice": float(np.mean([r["dice"] for r in records])),
    }
 
    name_w = max(len(n) for n in CATEGORY_ORDER)
    header = f"{'Category':<{name_w}} | {'Count':>5} | {'Mean IoU':>8} | {'Mean Dice':>9}"
    print("\nCategory breakdown (by ground-truth pedestrian area)\n")
    print(header)
    print("-" * len(header))
    for name in CATEGORY_ORDER:
        s = stats[name]
        iou_s = f"{s['mean_iou']:.4f}" if s["count"] else "n/a"
        dice_s = f"{s['mean_dice']:.4f}" if s["count"] else "n/a"
        print(f"{name:<{name_w}} | {s['count']:>5} | {iou_s:>8} | {dice_s:>9}")
    print("-" * len(header))
    print(f"{'Overall':<{name_w}} | {overall['count']:>5} | "
          f"{overall['mean_iou']:>8.4f} | {overall['mean_dice']:>9.4f}")
    empty = [n for n in CATEGORY_ORDER if stats[n]["count"] == 0]
    if empty:
        print(f"\nNote: no images fell into: {', '.join(empty)}.")
 
    
    k = max(1, min(args.top_k, len(records)))
    order = np.argsort([r["iou"] for r in records])      
    worst_idx = [int(i) for i in order[:k]]
    best_idx = [int(i) for i in order[::-1][:k]]
 
    artifacts_dir = resolve(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    for stale in list(artifacts_dir.glob("worst_*.png")) + list(artifacts_dir.glob("best_*.png")):
        stale.unlink()  
 
    def render(indices: List[int], tag: str) -> List[dict]:
        entries = []
        for rank, idx in enumerate(indices, start=1):
            image_t, gt_t = dataset[idx]
            image = denormalize(image_t, prep["mean"], prep["std"])
            out_path = artifacts_dir / f"{tag}_{rank:02d}_{Path(files[idx]).stem}_iou{records[idx]['iou']:.3f}.png"
            save_comparison(out_path, image, gt_t.numpy(), pred_masks[idx],
                            records[idx], files[idx], f"{tag.upper()} 
            entries.append(case_entry(rank, idx, files, records, out_path))
        return entries
 
    worst_cases = render(worst_idx, "worst")
    best_cases = render(best_idx, "best")
 
    print(f"\nWorst {k} images by IoU:")
    for c in worst_cases:
        print(f"  
              f"GT {c['gt_area_pct']:.1f}% vs pred {c['pred_area_pct']:.1f}% | {c['diagnosis']}")
    print(f"\nBest {k} images by IoU:")
    for c in best_cases:
        print(f"  
 
    
    flagged = [
        {
            "image": files[i],
            "iou": r["iou"],
            "iou_if_inverted": r["iou_if_inverted"],
            "gt_area_pct": r["gt_area_frac"] * 100.0,
            "pred_area_pct": r["pred_area_frac"] * 100.0,
        }
        for i, r in enumerate(records)
        if r["iou_if_inverted"] > r["iou"]
    ]
    print(f"\nInversion audit: {len(flagged)} of {len(records)} images would score higher "
          "with an inverted mask (diagnostic only; predictions were NOT modified).")
    for f_ in flagged[:10]:
        print(f"  {f_['image']}: IoU {f_['iou']:.3f} -> {f_['iou_if_inverted']:.3f} if inverted")
 
    
    report = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint": str(config["model"]["checkpoint_path"]),
            "device": str(device),
            "split": args.split,
            "num_samples": len(records),
            "threshold": threshold,
            "image_size": list(prep["image_size"]),
            "post_processing": "none (plain sigmoid > threshold)",
            "category_definitions": {
                SMALL: "GT area < 15% of frame",
                MEDIUM: "15% <= GT area <= 50% of frame",
                LARGE: "GT area > 50% of frame",
            },
        },
        "overall": overall,
        "categories": stats,
        "worst_cases": worst_cases,
        "best_cases": best_cases,
        "inversion_audit": {
            "num_flagged": len(flagged),
            "num_images": len(records),
            "note": "Flagged = inverted mask would have a higher IoU. Diagnostic only.",
            "flagged_images": flagged,
        },
    }
    report_path = resolve(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
 
    print(f"\nArtifacts saved to {artifacts_dir}")
    print(f"Report saved to {report_path}")
    return 0
 
 
if __name__ == "__main__":
    sys.exit(main())
