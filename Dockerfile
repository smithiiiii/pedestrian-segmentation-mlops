FROM python:3.10-slim-bookworm AS builder
 
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100
 
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
 
# Only requirements.txt is copied here, so editing application code never
# invalidates the (slow) dependency install layer.
COPY requirements.txt .
 
# No PyTorch in this image: inference runs on ONNX Runtime.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt
 
 
# ----------------------------------------------------------------------------
# Stage 2: slim runtime image
# ----------------------------------------------------------------------------
FROM python:3.10-slim-bookworm AS runtime
 
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    ORT_INTRA_OP_THREADS=0 \
    PATH="/opt/venv/bin:$PATH"
 
# Shared libraries OpenCV needs at import time on a slim base image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
 
# Run as an unprivileged user.
RUN groupadd --system --gid 10001 appuser \
    && useradd --system --uid 10001 --gid appuser --create-home --shell /usr/sbin/nologin appuser
 
WORKDIR /app
 
COPY --from=builder /opt/venv /opt/venv
 
# Application code and assets. Copying the model by exact filename makes the
# build fail immediately if best_model_b.onnx is missing, rather than at startup.
COPY app/ ./app/
COPY configs/ ./configs/
COPY models/best_model_b.onnx ./models/best_model_b.onnx
 
USER appuser
 
EXPOSE 8000
 
# python:slim ships without curl, so the probe uses the standard library.
# start-period leaves time for the model to load before failures count.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD ["python", "-c", "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health', timeout=3).read()"]
 
# One worker per container: each worker holds its own copy of the model in memory,
# so scale out by running more containers/replicas.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
 