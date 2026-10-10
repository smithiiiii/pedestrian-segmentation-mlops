import base64
import os
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from app.onnx_inference import InvalidImageError, OnnxSegmentationService, preprocess_image

ROOT = Path(__file__).resolve().parent.parent
ONNX_PATH = ROOT / "models" / "best_model_b.onnx"
PT_PATH = ROOT / "models" / "best_model_b.pt"

with open(ROOT / "configs" / "config.yaml", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

IMAGE_SIZE = tuple(int(v) for v in CONFIG["preprocessing"]["image_size"])
MEAN = tuple(CONFIG["preprocessing"]["mean"])
STD = tuple(CONFIG["preprocessing"]["std"])


def require_model() -> None:
    if ONNX_PATH.exists():
        return
    if os.environ.get("CI"):
        pytest.fail(f"Missing {ONNX_PATH}; commit it (Git LFS) so CI can test it.", pytrace=False)
    pytest.skip(f"{ONNX_PATH.name} not found; run training/export_onnx.py first.")


@pytest.fixture(scope="module")
def service() -> OnnxSegmentationService:
    require_model()
    return OnnxSegmentationService(model_path=ONNX_PATH)


@pytest.fixture(scope="module")
def sample_image() -> np.ndarray:
    return np.random.default_rng(0).integers(0, 256, size=(480, 640, 3), dtype=np.uint8)


def encode(image: np.ndarray, ext: str) -> bytes:
    ok, buffer = cv2.imencode(ext, image)
    assert ok
    return buffer.tobytes()


def decode_mask(mask_base64: str) -> np.ndarray:
    raw = np.frombuffer(base64.b64decode(mask_base64), dtype=np.uint8)
    mask = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    assert mask is not None
    return mask


def test_preprocess_shape_dtype_layout(sample_image):
    tensor = preprocess_image(sample_image, IMAGE_SIZE, MEAN, STD)
    assert tensor.shape == (1, 3, *IMAGE_SIZE)
    assert tensor.dtype == np.float32
    assert tensor.flags["C_CONTIGUOUS"]


def test_preprocess_matches_reference_formula(sample_image):
    resized = cv2.resize(sample_image, (IMAGE_SIZE[1], IMAGE_SIZE[0]), interpolation=cv2.INTER_LINEAR)
    scaled = resized.astype(np.float32) / 255.0
    expected = ((scaled - np.array(MEAN, np.float32)) / np.array(STD, np.float32)).transpose(2, 0, 1)[None]
    actual = preprocess_image(sample_image, IMAGE_SIZE, MEAN, STD)
    np.testing.assert_allclose(actual, expected, atol=1e-5)


def test_predict_response_contract(service, sample_image):
    result = service.predict(encode(sample_image, ".png"))
    assert set(result) == {"mask_base64", "inference_time_ms", "original_shape"}
    assert result["original_shape"] == (480, 640)
    assert result["inference_time_ms"] > 0
    mask = decode_mask(result["mask_base64"])
    assert mask.shape == IMAGE_SIZE
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)) <= {0, 255}


def test_predict_is_deterministic(service, sample_image):
    payload = encode(sample_image, ".png")
    first = service.predict(payload)["mask_base64"]
    second = service.predict(payload)["mask_base64"]
    assert first == second


def test_predict_accepts_png_and_jpeg(service, sample_image):
    for ext in (".png", ".jpg"):
        result = service.predict(encode(sample_image, ext))
        assert result["original_shape"] == (480, 640)
        assert decode_mask(result["mask_base64"]).shape == IMAGE_SIZE


@pytest.mark.parametrize(
    "payload",
    [b"", b"not an image", b"\x89PNG\r\n\x1a\n" + b"\x00" * 16],
)
def test_invalid_payload_raises(service, payload):
    with pytest.raises(InvalidImageError):
        service.predict(payload)


def test_oversized_payload_raises():
    require_model()
    small = OnnxSegmentationService(model_path=ONNX_PATH, max_image_bytes=16)
    with pytest.raises(InvalidImageError):
        small.predict(b"x" * 17)


def test_onnx_matches_pytorch(service, sample_image):
    torch = pytest.importorskip("torch")
    smp = pytest.importorskip("segmentation_models_pytorch")
    if not PT_PATH.exists():
        pytest.skip("PyTorch weights not available")

    model_cfg = CONFIG["model"]
    model = smp.Unet(
        encoder_name=model_cfg["encoder_name"],
        encoder_weights=None,
        in_channels=model_cfg["in_channels"],
        classes=model_cfg["classes"],
    )
    model.load_state_dict(torch.load(PT_PATH, map_location="cpu"))
    model.eval()

    tensor = preprocess_image(sample_image, IMAGE_SIZE, MEAN, STD)
    with torch.no_grad():
        expected = model(torch.from_numpy(tensor)).numpy()
    actual = service.session.run([service.output_name], {service.input_name: tensor})[0]

    assert np.abs(actual - expected).max() < 1e-3
    assert ((actual > 0) == (expected > 0)).mean() > 0.999
