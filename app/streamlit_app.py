import base64
import binascii
import io
import os

import numpy as np
import requests
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

API_URL = os.environ.get("SEGMENTATION_API_URL", "http://127.0.0.1:8000/predict")
CONNECT_TIMEOUT_S = 3
READ_TIMEOUT_S = 60
CONTENT_TYPES = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}
REQUIRED_KEYS = ("mask_base64", "inference_time_ms", "original_shape")
OVERLAY_COLOR = (255, 0, 0)


class ApiError(Exception):
    """Raised for any failure while preparing the request or handling the API response."""


def extract_error_detail(response: requests.Response) -> str:
    try:
        detail = response.json().get("detail", response.text)
    except (ValueError, AttributeError):
        return response.text or response.reason
    if isinstance(detail, list):
        return "; ".join(str(item.get("msg", item)) if isinstance(item, dict) else str(item) for item in detail)
    return str(detail)


@st.cache_data(show_spinner=False)
def fetch_prediction(image_bytes: bytes, filename: str, content_type: str) -> dict:
    try:
        response = requests.post(
            API_URL,
            files={"file": (filename, image_bytes, content_type)},
            timeout=(CONNECT_TIMEOUT_S, READ_TIMEOUT_S),
        )
    except requests.exceptions.ConnectionError as exc:
        raise ApiError(f"Cannot reach the API at {API_URL}. Is the container running on port 8000?") from exc
    except requests.exceptions.Timeout as exc:
        raise ApiError(f"The API did not respond within {READ_TIMEOUT_S} seconds.") from exc
    except requests.exceptions.RequestException as exc:
        raise ApiError(f"Request failed: {exc}") from exc

    if response.status_code != 200:
        raise ApiError(f"API error {response.status_code}: {extract_error_detail(response)}")

    try:
        payload = response.json()
    except ValueError as exc:
        raise ApiError("The API returned a response that is not valid JSON.") from exc

    if not isinstance(payload, dict):
        raise ApiError("The API returned an unexpected response format.")
    missing = [key for key in REQUIRED_KEYS if key not in payload]
    if missing:
        raise ApiError(f"The API response is missing fields: {', '.join(missing)}.")
    return payload


def load_image(image_bytes: bytes) -> Image.Image:
    try:
        image = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes)))
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ApiError("The uploaded file is not a valid image.") from exc


def decode_mask(mask_base64: str, size: tuple[int, int]) -> np.ndarray:
    try:
        raw = base64.b64decode(mask_base64, validate=True)
        mask = Image.open(io.BytesIO(raw)).convert("L")
    except (binascii.Error, ValueError, UnidentifiedImageError, OSError) as exc:
        raise ApiError("Could not decode the mask returned by the API.") from exc
    mask = mask.resize(size, Image.Resampling.NEAREST)
    return np.asarray(mask) > 0


def blend_overlay(image: Image.Image, mask: np.ndarray, opacity: float) -> Image.Image:
    pixels = np.asarray(image, dtype=np.float32)
    alpha = mask[..., np.newaxis].astype(np.float32) * opacity
    color = np.asarray(OVERLAY_COLOR, dtype=np.float32)
    blended = pixels * (1.0 - alpha) + color * alpha
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


def main() -> None:
    st.set_page_config(page_title="Pedestrian Segmentation", page_icon="🚶", layout="wide")
    st.title("🚶 Pedestrian Segmentation")
    st.caption(f"U-Net (ResNet34) on ONNX Runtime · endpoint: {API_URL}")

    uploaded = st.file_uploader("Upload an image", type=list(CONTENT_TYPES))
    opacity = st.slider("Overlay opacity", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

    if uploaded is None:
        st.info("Upload a PNG or JPEG image to get started.")
        return

    image_bytes = uploaded.getvalue()
    extension = uploaded.name.rsplit(".", 1)[-1].lower()
    content_type = CONTENT_TYPES.get(extension, uploaded.type or "application/octet-stream")

    try:
        image = load_image(image_bytes)
    except ApiError as exc:
        st.error(str(exc))
        return

    try:
        with st.spinner("Running segmentation..."):
            result = fetch_prediction(image_bytes, uploaded.name, content_type)
        mask = decode_mask(result["mask_base64"], image.size)
    except ApiError as exc:
        st.error(str(exc))
        st.image(image, caption="Uploaded image", width="stretch")
        return

    st.success(f"Inference complete in {float(result['inference_time_ms']):.1f} ms!")

    original_col, overlay_col = st.columns(2)
    with original_col:
        st.subheader("Original")
        st.image(image, width="stretch")
    with overlay_col:
        st.subheader("Segmentation overlay")
        st.image(blend_overlay(image, mask, opacity), width="stretch")

    coverage = float(mask.mean()) * 100
    st.caption(f"Original shape (H×W): {tuple(result['original_shape'])} · pedestrian coverage: {coverage:.1f}%")


main()
