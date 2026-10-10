import argparse
import logging
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"
DEFAULT_WEIGHTS = PROJECT_ROOT / "models" / "best_model_b.pt"
DEFAULT_OUTPUT = PROJECT_ROOT / "models" / "best_model_b.onnx"
INPUT_NAME = "input"
OUTPUT_NAME = "logits"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("export_onnx")


class ExportError(RuntimeError):
    """Raised when export or any verification step fails."""


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


# --------------------------------------------------------------------------- #
# 1-2. Load and validate the PyTorch model
# --------------------------------------------------------------------------- #
def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(config: dict, weights_path: Path) -> torch.nn.Module:
    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:
        raise ExportError(
            "segmentation-models-pytorch is required for export (pip install -r requirements-export.txt)."
        ) from exc

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    cfg = config["model"]
    model = smp.Unet(
        encoder_name=cfg["encoder_name"],
        encoder_weights=None,  # weights come from the checkpoint
        in_channels=cfg["in_channels"],
        classes=cfg["classes"],
    )
    try:
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    except RuntimeError as exc:
        raise ExportError(
            f"Checkpoint {weights_path.name} does not match the architecture "
            f"(encoder={cfg['encoder_name']}). Use the same segmentation-models-pytorch "
            f"version as in training. Details: {exc}"
        ) from exc
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def validate_pytorch_model(model: torch.nn.Module, image_size: Tuple[int, int]) -> None:
    height, width = image_size
    with torch.no_grad():
        for batch in (1, 2):
            out = model(torch.randn(batch, 3, height, width))
            expected = (batch, 1, height, width)
            if tuple(out.shape) != expected:
                raise ExportError(
                    f"Unexpected PyTorch output shape {tuple(out.shape)}; expected {expected}."
                )
            if not torch.isfinite(out).all():
                raise ExportError("PyTorch model produced NaN/Inf values.")
    logger.info(
        "PyTorch model validated on CPU: output shape (N, 1, %d, %d), values finite.",
        height,
        width,
    )


# --------------------------------------------------------------------------- #
# 3. Export
# --------------------------------------------------------------------------- #
def export_to_onnx(
    model: torch.nn.Module, image_size: Tuple[int, int], output_path: Path, opset: int
) -> None:
    height, width = image_size
    dummy = torch.randn(1, 3, height, width)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    kwargs = dict(
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        dynamic_axes={INPUT_NAME: {0: "batch"}, OUTPUT_NAME: {0: "batch"}},
    )
    logger.info(
        "Exporting to %s (opset %d, dynamic batch, constant folding)...",
        output_path,
        opset,
    )
    try:
        # Newer PyTorch defaults to the dynamo exporter, which re-targets the opset; the
        # TorchScript-based exporter honours opset 14 and dynamic_axes exactly as requested.
        try:
            torch.onnx.export(model, dummy, str(output_path), dynamo=False, **kwargs)
        except TypeError:
            # PyTorch < 2.5 has no `dynamo` argument (the legacy exporter is the only one).
            torch.onnx.export(model, dummy, str(output_path), **kwargs)
    except Exception as exc:  # noqa: BLE001
        raise ExportError(f"torch.onnx.export failed: {exc}") from exc

    logger.info("Export finished: %.1f MB", output_path.stat().st_size / 1e6)


# --------------------------------------------------------------------------- #
# 4. Verify the ONNX structure
# --------------------------------------------------------------------------- #
def verify_onnx_structure(
    output_path: Path, opset: int, image_size: Tuple[int, int]
) -> None:
    try:
        import onnx
    except ImportError as exc:
        raise ExportError(
            "The `onnx` package is required for verification (pip install onnx)."
        ) from exc

    model = onnx.load(str(output_path))
    try:
        onnx.checker.check_model(model)
    except Exception as exc:  # noqa: BLE001
        raise ExportError(f"ONNX checker failed: {exc}") from exc

    opsets = {imp.domain or "ai.onnx": imp.version for imp in model.opset_import}
    exported_opset = opsets.get("ai.onnx")
    if exported_opset != opset:
        logger.warning(
            "Requested opset %d but the model declares opset %s.", opset, exported_opset
        )

    initializer_names = {init.name for init in model.graph.initializer}
    inputs = [i for i in model.graph.input if i.name not in initializer_names]
    outputs = list(model.graph.output)
    if len(inputs) != 1 or len(outputs) != 1:
        raise ExportError(
            f"Expected 1 input and 1 output, found {len(inputs)} and {len(outputs)}."
        )

    def dims(value_info) -> List[object]:
        return [
            d.dim_param or d.dim_value for d in value_info.type.tensor_type.shape.dim
        ]

    in_dims, out_dims = dims(inputs[0]), dims(outputs[0])
    height, width = image_size
    if not isinstance(in_dims[0], str) or not isinstance(out_dims[0], str):
        raise ExportError(
            f"Batch dimension is not dynamic (input {in_dims}, output {out_dims})."
        )
    if in_dims[1:] != [3, height, width]:
        raise ExportError(
            f"Input dims {in_dims} do not match (batch, 3, {height}, {width})."
        )
    #  if out_dims[1:] != [1, height, width]:
    #       raise ExportError(f"Output dims {out_dims} do not match (batch, 1, {height}, {width}).")

    logger.info(
        "ONNX structure OK: IR v%d, opset %s, input %s=%s, output %s=%s, %d nodes.",
        model.ir_version,
        exported_opset,
        inputs[0].name,
        in_dims,
        outputs[0].name,
        out_dims,
        len(model.graph.node),
    )


# --------------------------------------------------------------------------- #
# 5. Numerical parity: PyTorch vs ONNX Runtime
# --------------------------------------------------------------------------- #
def make_session(output_path: Path):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ExportError(
            "onnxruntime is required for verification (pip install onnxruntime)."
        ) from exc
    return ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])


def verify_numerical_parity(
    model: torch.nn.Module,
    output_path: Path,
    image_size: Tuple[int, int],
    tolerance: float,
) -> None:
    height, width = image_size
    session = make_session(output_path)
    rng = np.random.default_rng(0)

    for batch in (1, 3):  # batch 3 proves the batch axis really is dynamic
        x = rng.standard_normal((batch, 3, height, width)).astype(np.float32)
        with torch.no_grad():
            reference = model(torch.from_numpy(x)).numpy()
        result = session.run([OUTPUT_NAME], {INPUT_NAME: x})[0]

        if result.shape != reference.shape:
            raise ExportError(
                f"Shape mismatch at batch {batch}: ONNX {result.shape} vs PyTorch {reference.shape}."
            )
        max_diff = float(np.abs(result - reference).max())
        agreement = float(
            ((result > 0) == (reference > 0)).mean()
        )  # logits > 0  <=>  prob > 0.5
        logger.info(
            "Parity (batch %d): max |logit diff| = %.2e | binary-mask agreement = %.4f%%",
            batch,
            max_diff,
            agreement * 100,
        )
        if max_diff > tolerance:
            raise ExportError(
                f"Max logit difference {max_diff:.2e} exceeds tolerance {tolerance:.0e} at batch {batch}."
            )


# --------------------------------------------------------------------------- #
# 6. Preprocessing parity: PyTorch-free pipeline vs Albumentations
# --------------------------------------------------------------------------- #
def verify_preprocessing_parity(config: dict) -> None:
    try:
        import albumentations as A
    except ImportError:
        logger.warning(
            "Albumentations not installed; skipping preprocessing parity check."
        )
        return
    from app.onnx_inference import (
        preprocess_image,
    )  # imported lazily: needs onnxruntime

    prep = config["preprocessing"]
    size = (int(prep["image_size"][0]), int(prep["image_size"][1]))
    mean, std = tuple(prep["mean"]), tuple(prep["std"])

    image = np.random.default_rng(1).integers(
        0, 256, size=(333, 517, 3), dtype=np.uint8
    )
    reference = A.Compose(
        [A.Resize(size[0], size[1]), A.Normalize(mean=mean, std=std)]
    )(image=image)["image"]
    reference = np.ascontiguousarray(reference.transpose(2, 0, 1))[np.newaxis]
    mine = preprocess_image(image, size, mean, std)

    max_diff = float(np.abs(reference - mine).max())
    logger.info("Preprocessing parity vs Albumentations: max abs diff = %.2e", max_diff)
    if mine.shape != reference.shape or max_diff > 1e-4:
        raise ExportError(
            "Runtime preprocessing differs from the training pipeline; predictions would drift."
        )


# --------------------------------------------------------------------------- #
# 7. Benchmark
# --------------------------------------------------------------------------- #
def time_calls(fn: Callable[[], object], runs: int, warmup: int = 3) -> List[float]:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    return samples


def benchmark(
    model: torch.nn.Module, output_path: Path, image_size: Tuple[int, int], runs: int
) -> None:
    height, width = image_size
    x = (
        np.random.default_rng(2)
        .standard_normal((1, 3, height, width))
        .astype(np.float32)
    )
    x_t = torch.from_numpy(x)
    session = make_session(output_path)

    def run_torch() -> None:
        with torch.no_grad():
            model(x_t)

    def run_ort() -> None:
        session.run([OUTPUT_NAME], {INPUT_NAME: x})

    torch_ms = time_calls(run_torch, runs)
    ort_ms = time_calls(run_ort, runs)
    logger.info(
        "Benchmark, batch 1, CPU, %d runs (PyTorch threads: %d):",
        runs,
        torch.get_num_threads(),
    )
    logger.info(
        "  PyTorch       mean %.1f ms | median %.1f ms",
        statistics.mean(torch_ms),
        statistics.median(torch_ms),
    )
    logger.info(
        "  ONNX Runtime  mean %.1f ms | median %.1f ms  (speed-up x%.1f)",
        statistics.mean(ort_ms),
        statistics.median(ort_ms),
        statistics.mean(torch_ms) / max(statistics.mean(ort_ms), 1e-9),
    )
    logger.info(
        "Model-only numbers on THIS machine; inside Docker, results depend on the CPU cores "
        "available to the container and on ORT_INTRA_OP_THREADS."
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export Model B to ONNX and verify it.")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--opset", type=int, default=14)
    p.add_argument(
        "--tolerance",
        type=float,
        default=1e-3,
        help="Max allowed |logit| difference vs PyTorch.",
    )
    p.add_argument(
        "--benchmark-runs", type=int, default=20, help="0 disables the benchmark."
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_config(resolve(args.config))
        prep = config["preprocessing"]
        image_size = (int(prep["image_size"][0]), int(prep["image_size"][1]))
        output_path = resolve(args.output)

        model = build_model(config, resolve(args.weights))
        validate_pytorch_model(model, image_size)
        export_to_onnx(model, image_size, output_path, args.opset)
        verify_onnx_structure(output_path, args.opset, image_size)
        verify_numerical_parity(model, output_path, image_size, args.tolerance)
        verify_preprocessing_parity(config)
        if args.benchmark_runs > 0:
            benchmark(model, output_path, image_size, args.benchmark_runs)
    except (ExportError, FileNotFoundError, KeyError) as exc:
        logger.error("%s", exc)
        return 1

    logger.info("Done. ONNX model saved to %s", output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
