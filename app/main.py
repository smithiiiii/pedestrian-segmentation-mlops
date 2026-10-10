import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Tuple

# Make `app.inference` importable regardless of the directory the server is started from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import (
    Depends,
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
)  # noqa: E402
from fastapi.concurrency import run_in_threadpool  # noqa: E402
from pydantic import BaseModel  # noqa: E402

# ONNX Runtime service (no PyTorch at runtime). To fall back to the PyTorch service use:
#   from app.inference import InvalidImageError, SegmentationService
from app.onnx_inference import InvalidImageError  # noqa: E402
from app.onnx_inference import (
    OnnxSegmentationService as SegmentationService,
)  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("pedseg.api")

ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg"}


# --------------------------------------------------------------------------- #
# Response schemas
# --------------------------------------------------------------------------- #
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class PredictionResponse(BaseModel):
    mask_base64: str  # base64 PNG, single channel (0 = background, 255 = pedestrian)
    inference_time_ms: float  # preprocessing + forward pass + thresholding
    original_shape: Tuple[int, int]  # (height, width) of the uploaded image


# --------------------------------------------------------------------------- #
# Application lifecycle: load the model exactly once, at startup
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading segmentation model...")
    app.state.service = (
        SegmentationService()
    )  # a failure here aborts startup with a clear error
    logger.info("Model loaded; API is ready.")
    yield
    app.state.service = None
    logger.info("Shutdown complete.")


app = FastAPI(
    title="Pedestrian Segmentation API",
    description="U-Net (ResNet34) binary pedestrian segmentation.",
    version="1.0.0",
    lifespan=lifespan,
)


def get_service(request: Request) -> SegmentationService:
    """Dependency: the shared, already-loaded model service."""
    service = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    return service


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    loaded = getattr(request.app.state, "service", None) is not None
    return HealthResponse(
        status="healthy" if loaded else "unhealthy", model_loaded=loaded
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    service: SegmentationService = Depends(get_service),
) -> PredictionResponse:
    # 1) Validate the declared content type (parameters such as "; charset=..." are ignored)
    content_type = (file.content_type or "").split(";")[0].strip().lower()
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content type '{file.content_type}'. Upload a PNG or JPEG image.",
        )

    # 2) Read at most limit + 1 bytes so an oversized upload can't exhaust memory;
    #    the service rejects anything over its limit with InvalidImageError.
    image_bytes = await file.read(service.max_image_bytes + 1)

    # 3) Run the blocking model call off the event loop so /health stays responsive
    try:
        result = await run_in_threadpool(service.predict, image_bytes)
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:  # noqa: BLE001 - never leak internals to clients
        logger.exception("Inference failed for upload '%s'", file.filename)
        raise HTTPException(status_code=500, detail="Internal error during inference.")

    logger.info(
        "Predicted '%s' (%sx%s) in %.1f ms",
        file.filename,
        result["original_shape"][1],
        result["original_shape"][0],
        result["inference_time_ms"],
    )
    return PredictionResponse(**result)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)
