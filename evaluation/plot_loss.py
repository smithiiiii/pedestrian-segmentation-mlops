import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except ImportError:
    sns = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HISTORY_PATH = PROJECT_ROOT / "training" / "history_model_b.json"
OUTPUT_PATH = PROJECT_ROOT / "media" / "loss_curve.png"
TITLE = "Model B (ResNet34 U-Net) Training Loss"
DPI = 300


def load_history(path: Path) -> tuple:
    if not path.exists():
        raise FileNotFoundError(
            f"History file not found: {path}. Train Model B first (python training/train_model_b.py)."
        )
    with open(path, "r", encoding="utf-8") as f:
        history = json.load(f)

    if "train_loss" not in history or not history["train_loss"]:
        raise KeyError(f"'train_loss' is missing or empty in {path}")
    losses = [float(v) for v in history["train_loss"]]
    epochs = history.get("epoch") or list(range(1, len(losses) + 1))
    if len(epochs) != len(losses):
        raise ValueError(f"'epoch' ({len(epochs)}) and 'train_loss' ({len(losses)}) lengths differ.")
    return epochs, losses


def apply_style() -> None:
    if sns is not None:
        sns.set_theme(style="whitegrid", context="talk")
    else:
        plt.style.use("ggplot")


def main() -> int:
    try:
        epochs, losses = load_history(HISTORY_PATH)
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    apply_style()
    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(
        epochs,
        losses,
        marker="o",
        markersize=5,
        linewidth=2.2,
        color="#2b5c8f",
        label="Train loss (BCE + Dice)",
    )

    final_epoch, final_loss = epochs[-1], losses[-1]
    ax.scatter(
        [final_epoch],
        [final_loss],
        s=140,
        color="#d95f02",
        edgecolor="white",
        linewidth=1.5,
        label=f"Final loss: {final_loss:.4f}",
    )
    ax.annotate(
        f"Final: {final_loss:.4f}",
        xy=(final_epoch, final_loss),
        xytext=(-70, 40),
        textcoords="offset points",
        fontsize=12,
        fontweight="bold",
        color="#d95f02",
        arrowprops=dict(arrowstyle="->", color="#d95f02"),
    )

    ax.set_title(TITLE, fontsize=16, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Loss", fontsize=13)
    if len(epochs) <= 25:
        ax.set_xticks(epochs)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper right", fontsize=11)

    fig.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=DPI)
    plt.close(fig)

    print(f"Loss curve saved to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
