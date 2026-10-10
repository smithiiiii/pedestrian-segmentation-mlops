import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response

from src.inference import InferenceEngine

app = FastAPI(title="Pedestrian Segmentation API", version="1.0.0")
engine = InferenceEngine()


@app.get("/health")
async def health() -> dict:
    return {"status": "healthy"}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    threshold: float = Query(0.50, ge=0.0, le=1.0),
) -> Response:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid content type '{file.content_type}'. Please upload an image.",
        )

    image_bytes = await file.read()

    try:
        mask = await run_in_threadpool(engine.predict, image_bytes, threshold)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

    success, buffer = cv2.imencode(".png", mask)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode mask as PNG.")

    return Response(content=buffer.tobytes(), media_type="image/png")
