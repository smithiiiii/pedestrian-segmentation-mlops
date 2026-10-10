import argparse
import csv
import random
import sys
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

try:
    from pycocotools.coco import COCO
except ImportError:
    COCO = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANNOTATIONS_ZIP_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
BIN_ORDER = ("small", "medium", "large")
MANIFEST_COLUMNS = [
    "image_path",
    "mask_path",
    "source",
    "size_bin",
    "person_area_frac",
    "width",
    "height",
    "source_id",
]


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def rel_posix(path: Path) -> str:
    """Project-relative path with forward slashes (portable across Windows/Linux)."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def size_bin(frac: float) -> str:
    """Same thresholds as evaluation/error_analysis.py."""
    if frac < 0.15:
        return "small"
    if frac <= 0.50:
        return "medium"
    return "large"


def download_file(url: str, dest: Path, retries: int = 3, timeout: int = 60, show_progress: bool = False) -> None:
    """Download to a temp file, then atomically move into place."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    last_exc: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp, open(tmp, "wb") as out:
                total = int(resp.headers.get("Content-Length") or 0)
                done = 0
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
                    done += len(chunk)
                    if show_progress and total:
                        print(
                            f"\r  {done / 1e6:.0f}/{total / 1e6:.0f} MB ({100 * done / total:.0f}%)", end="", flush=True
                        )
            if show_progress:
                print()
            tmp.replace(dest)
            return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_exc = exc
            time.sleep(2 * attempt)
    if tmp.exists():
        tmp.unlink()
    raise RuntimeError(f"Failed to download {url}: {last_exc}")


def ensure_annotations(split: str, ann_file: Optional[Path], cache_dir: Path) -> Path:
    if ann_file is not None:
        ann_file = resolve(ann_file)
        if not ann_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {ann_file}")
        return ann_file

    json_path = cache_dir / f"instances_{split}.json"
    if json_path.exists():
        return json_path

    zip_path = cache_dir / "annotations_trainval2017.zip"
    if not zip_path.exists():
        print(f"Downloading COCO annotations (~250 MB) from {ANNOTATIONS_ZIP_URL}")
        download_file(ANNOTATIONS_ZIP_URL, zip_path, retries=3, timeout=120, show_progress=True)

    member = f"annotations/instances_{split}.json"
    print(f"Extracting {member}")
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member) as src, open(json_path, "wb") as dst:
            while True:
                chunk = src.read(1 << 20)
                if not chunk:
                    break
                dst.write(chunk)
    zip_path.unlink()
    return json_path


def person_mask(coco: "COCO", img_info: dict, cat_ids: List[int]) -> np.ndarray:
    """Union of all person annotations (polygon or RLE) as a 0/1 uint8 array (H, W)."""
    height, width = img_info["height"], img_info["width"]
    mask = np.zeros((height, width), dtype=np.uint8)
    ann_ids = coco.getAnnIds(imgIds=img_info["id"], catIds=cat_ids, iscrowd=None)
    for ann in coco.loadAnns(ann_ids):
        try:
            mask |= coco.annToMask(ann).astype(np.uint8)
        except Exception:
            continue
    return mask


def allocate(available: Dict[str, int], total: int) -> Dict[str, int]:
    """Split `total` equally across bins; redistribute shortfalls of small bins."""
    quota = {b: 0 for b in BIN_ORDER}
    remaining = total
    active = [b for b in BIN_ORDER if available[b] > 0]
    while remaining > 0 and active:
        share, extra = divmod(remaining, len(active))
        taken = 0
        for i, b in enumerate(active):
            want = share + (1 if i < extra else 0)
            take = min(want, available[b] - quota[b])
            quota[b] += take
            taken += take
        remaining -= taken
        active = [b for b in active if quota[b] < available[b]]
        if taken == 0:
            break
    return quota


def process_image(
    coco: "COCO", img_info: dict, cat_ids: List[int], images_dir: Path, masks_dir: Path
) -> Optional[dict]:
    """Download one image, write its binary mask, return a manifest record (or None)."""
    file_name = img_info["file_name"]
    img_path = images_dir / file_name
    mask_path = masks_dir / f"{Path(file_name).stem}.png"
    try:
        if not img_path.exists():
            download_file(img_info["coco_url"], img_path)
        with Image.open(img_path) as im:
            im.verify()
        with Image.open(img_path) as im:
            width, height = im.size

        mask = person_mask(coco, img_info, cat_ids)
        if mask.shape != (height, width):
            mask = np.array(Image.fromarray(mask).resize((width, height), Image.NEAREST))
        binary = (mask > 0).astype(np.uint8)
        masks_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(binary * 255, mode="L").save(mask_path)

        frac = float(binary.mean())
        return {
            "image_path": rel_posix(img_path),
            "mask_path": rel_posix(mask_path),
            "source": "coco",
            "size_bin": size_bin(frac),
            "person_area_frac": round(frac, 6),
            "width": width,
            "height": height,
            "source_id": img_info["id"],
        }
    except Exception as exc:
        print(f"  [warn] skipping image {img_info.get('id')}: {exc}")
        for p in (img_path, mask_path):
            if p.exists():
                p.unlink()
        return None


def build_coco_subset(args: argparse.Namespace, images_dir: Path, masks_dir: Path) -> List[dict]:
    ann_dir = resolve(args.output_dir) / "annotations_cache"
    ann_path = ensure_annotations(args.coco_split, args.ann_file, ann_dir)

    print(f"Loading annotations: {ann_path}")
    coco = COCO(str(ann_path))
    cat_ids = coco.getCatIds(catNms=["person"])
    img_ids = sorted(coco.getImgIds(catIds=cat_ids))
    print(f"Images containing 'person': {len(img_ids)}")

    rng = random.Random(args.seed)
    rng.shuffle(img_ids)
    img_ids = img_ids[: args.max_candidates]

    print(f"Measuring person coverage for {len(img_ids)} candidate images...")
    bins: Dict[str, List[int]] = {b: [] for b in BIN_ORDER}
    for n, img_id in enumerate(img_ids, start=1):
        info = coco.loadImgs(img_id)[0]
        frac = float(person_mask(coco, info, cat_ids).mean())
        if frac >= args.min_area_frac:
            bins[size_bin(frac)].append(img_id)
        if n % 500 == 0:
            print(f"  {n}/{len(img_ids)}")

    available = {b: len(bins[b]) for b in BIN_ORDER}
    print("Eligible images per bin:", available)
    total_available = sum(available.values())
    if total_available < args.num_images:
        print(
            f"[warn] only {total_available} eligible images; requested {args.num_images}. "
            "Increase --max-candidates or use --coco-split train2017."
        )
    quota = allocate(available, args.num_images)
    print("Target per bin:", quota)

    records: List[dict] = []
    for b in BIN_ORDER:
        candidates = bins[b]
        selected: List[dict] = []
        pos = 0
        while len(selected) < quota[b] and pos < len(candidates):
            need = quota[b] - len(selected)
            batch = candidates[pos : pos + need]
            pos += need
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [
                    pool.submit(process_image, coco, coco.loadImgs(i)[0], cat_ids, images_dir, masks_dir) for i in batch
                ]
                for fut in futures:
                    rec = fut.result()
                    if rec is not None:
                        selected.append(rec)
            print(f"  [{b}] {len(selected)}/{quota[b]} ready")
        records.extend(selected)
    return records


def build_penn_records(penn_root: Path, binary_mask_dir: Path, val_split: float, seed: int) -> List[dict]:
    """Penn-Fudan TRAIN split (same split rule as the Colab script) with 0/255 masks."""
    img_dir, inst_mask_dir = penn_root / "PNGImages", penn_root / "PedMasks"
    if not img_dir.is_dir() or not inst_mask_dir.is_dir():
        raise FileNotFoundError(
            f"Penn-Fudan not found at {penn_root} (expected PNGImages/ and PedMasks/). "
            "Use --penn-root to point at the unzipped PennFudanPed folder."
        )
    files = sorted(f.name for f in img_dir.iterdir() if f.suffix.lower() == ".png")
    shuffled = files[:]
    random.Random(seed).shuffle(shuffled)
    n_val = int(len(shuffled) * val_split)
    train_files = sorted(shuffled[n_val:])

    binary_mask_dir.mkdir(parents=True, exist_ok=True)
    records: List[dict] = []
    for name in train_files:
        inst = np.array(Image.open(inst_mask_dir / name.replace(".png", "_mask.png")))
        binary = (inst > 0).astype(np.uint8)
        out_path = binary_mask_dir / name.replace(".png", "_mask.png")
        Image.fromarray(binary * 255, mode="L").save(out_path)

        with Image.open(img_dir / name) as im:
            width, height = im.size
        frac = float(binary.mean())
        records.append(
            {
                "image_path": rel_posix(img_dir / name),
                "mask_path": rel_posix(out_path),
                "source": "pennfudan",
                "size_bin": size_bin(frac),
                "person_area_frac": round(frac, 6),
                "width": width,
                "height": height,
                "source_id": Path(name).stem,
            }
        )
    return records


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare COCO person subset + unified manifest.")
    p.add_argument("--num-images", type=int, default=1000)
    p.add_argument(
        "--coco-split",
        choices=["val2017", "train2017"],
        default="val2017",
        help="val2017 annotations are small (~25 MB). train2017 needs ~2-3 GB RAM to load.",
    )
    p.add_argument(
        "--ann-file", type=Path, default=None, help="Use an existing instances_*.json instead of downloading."
    )
    p.add_argument(
        "--min-area-frac",
        type=float,
        default=0.005,
        help="Skip images where people cover less than this fraction of the frame.",
    )
    p.add_argument(
        "--max-candidates",
        type=int,
        default=6000,
        help="Cap on shuffled person images inspected (keeps train2017 runs fast).",
    )
    p.add_argument("--workers", type=int, default=8, help="Parallel image downloads.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, default=Path("data/coco_subset"))
    p.add_argument("--penn-root", type=Path, default=Path("data/PennFudanPed"))
    p.add_argument(
        "--penn-binary-dir",
        type=Path,
        default=Path("data/penn_fudan_binary/masks"),
        help="Where Penn-Fudan masks are re-saved as 0/255 binary PNGs.",
    )
    p.add_argument("--val-split", type=float, default=0.2, help="Penn-Fudan val fraction; must match training.")
    p.add_argument("--penn-seed", type=int, default=42, help="Penn-Fudan split seed; must match training.")
    p.add_argument("--manifest", type=Path, default=Path("data/combined_train_manifest.csv"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if COCO is None:
        print("pycocotools is not installed. Run: pip install pycocotools", file=sys.stderr)
        return 1
    if args.num_images < 1:
        print("--num-images must be >= 1", file=sys.stderr)
        return 1

    out_dir = resolve(args.output_dir)
    images_dir, masks_dir = out_dir / "images", out_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    try:
        penn_records = build_penn_records(
            resolve(args.penn_root), resolve(args.penn_binary_dir), args.val_split, args.penn_seed
        )
        print(f"Penn-Fudan training samples: {len(penn_records)}")
        coco_records = build_coco_subset(args, images_dir, masks_dir)
    except (FileNotFoundError, RuntimeError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    records = penn_records + coco_records
    manifest_path = resolve(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(records)

    print("\nManifest summary")
    print(f"{'source':<10} | {'small':>6} | {'medium':>6} | {'large':>6} | {'total':>6}")
    print("-" * 46)
    for source in ("pennfudan", "coco"):
        subset = [r for r in records if r["source"] == source]
        per_bin = [sum(1 for r in subset if r["size_bin"] == b) for b in BIN_ORDER]
        print(f"{source:<10} | {per_bin[0]:>6} | {per_bin[1]:>6} | {per_bin[2]:>6} | {len(subset):>6}")
    print("-" * 46)
    print(f"{'all':<10} | {'':>6} | {'':>6} | {'':>6} | {len(records):>6}")
    print(f"\nManifest saved to {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
