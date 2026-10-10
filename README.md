# Pedestrian Segmentation MLOps

**Optimized semantic segmentation. Low-latency CPU inference. Automated deployment workflows.**

`PyTorch` · `U-Net` · `ONNX Runtime` · `FastAPI` · `Docker` · `GitHub Actions`

[Overview](#overview) · [Architecture](#architecture) · [Performance](#performance) · [API](#api-reference) · [Quickstart](#quickstart) · [CI/CD](#cicd-pipeline)

---

## Overview

**pedestrian-segmentation-mlops** is an end-to-end machine learning engineering project that develops and serves a pedestrian segmentation model through a containerized inference API.

The system uses a U-Net architecture trained with PyTorch, exports the model to ONNX for optimized inference, and integrates ONNX Runtime into a FastAPI service. Docker provides a consistent execution environment, while GitHub Actions automates testing, code quality checks, and container image builds.

Rather than treating model training as the final deliverable, this project focuses on the engineering work required to make a computer vision model reproducible, testable, and accessible through a service interface.

### Key results

| Engineering dimension | Implementation |
|---|---|
| Computer vision | Pixel-level pedestrian segmentation using U-Net |
| Model optimization | ONNX export and ONNX Runtime inference |
| CPU performance | Approximately **71 ms inference latency** in the reported setup |
| Model serving | FastAPI prediction and health endpoints |
| Deployment packaging | Docker containerization |
| Continuous integration | Three-stage GitHub Actions workflow |
| Quality assurance | PyTest and Ruff |

> **Benchmark scope:** The approximately 71 ms figure represents the reported CPU inference latency. Hardware specifications, input dimensions, preprocessing overhead, warm-up procedure, and measurement methodology should be documented before treating it as a reproducible end-to-end service latency or making comparisons with other models.

---

## Engineering & MLOps Highlights

### 1. Model optimization and inference

- **PyTorch U-Net:** Implements an encoder-decoder architecture for pixel-level semantic segmentation.
- **ONNX deployment:** Separates the serving artifact from the training framework by exporting the trained model to the ONNX format.
- **ONNX Runtime:** Executes the exported model with a CPU-oriented inference path.
- **Latency awareness:** Targets approximately 71 ms CPU inference latency, with optimization focused on serving efficiency rather than training performance alone.
- **Inference consistency:** The preprocessing, tensor layout, normalization, output interpretation, and mask postprocessing must remain consistent between training and deployment.

### 2. API-first model serving

- Exposes model inference through a FastAPI REST interface.
- Provides a health endpoint for basic service availability checks.
- Uses Docker to package the application and its runtime dependencies.
- Exposes interactive API documentation through Swagger UI.
- Keeps the model execution layer separate from HTTP request handling to support modular testing and maintainability.

### 3. Automated quality gates

The GitHub Actions workflow implements three stages:

1. **Unit testing — PyTest:** Validates application and inference components through automated tests.
2. **Code quality — Ruff:** Checks Python code for configured linting violations.
3. **Container build — Docker:** Verifies that the application can be packaged into a container image.

These stages establish a repeatable validation process before an image is considered buildable. A successful image build does not, by itself, establish that the image has been deployed to production.

### 4. Reproducible execution

- Python 3.11+ development environment.
- Version-controlled source code, tests, and automation workflows.
- Container-based local execution.
- Explicit separation between training artifacts and inference-serving code.

---

## Architecture

```text
                    MODEL DEVELOPMENT
                            |
                            v
                   +------------------+
                   | Training Pipeline|
                   | PyTorch + U-Net  |
                   +------------------+
                            |
                            v
                   +------------------+
                   | Trained Model    |
                   +------------------+
                            |
                            v
                   +------------------+
                   | ONNX Export      |
                   +------------------+
                            |
                            v
                   +------------------+
                   | ONNX Runtime     |
                   | CPU Inference    |
                   +------------------+
                            |
                            v
+-------------+     +------------------+
| API Client  |---->| FastAPI Service  |
| Image Input |     | Predict / Health |
+-------------+     +------------------+
                            |
                            v
                   +------------------+
                   | Segmentation Mask |
                   +------------------+

                    DELIVERY WORKFLOW
                            |
                            v
                   +------------------+
                   | GitHub Actions   |
                   +------------------+
                            |
              +-------------+-------------+
              |             |             |
              v             v             v
          +-------+     +-------+     +---------+
          | PyTest|     | Ruff  |     | Docker  |
          | Tests |     | Lint  |     | Build   |
          +-------+     +-------+     +---------+
```

### Request lifecycle

1. A client submits an image to the prediction endpoint.
2. The API validates the request and applies the configured image preprocessing.
3. The inference layer converts the input into the tensor representation expected by the ONNX model.
4. ONNX Runtime executes the model.
5. The application interprets the output and produces a segmentation mask in the documented response format.
6. The API returns the result to the client.

The exact response representation—such as a JSON-encoded mask, an encoded image, or a downloadable artifact—depends on the implementation.

---

## Technology Stack

| Category | Technologies | Responsibility |
|---|---|---|
| Core ML | Python 3.11+, PyTorch, U-Net | Model development and training |
| Model deployment | ONNX | Portable model representation |
| Serving & inference | ONNX Runtime, FastAPI | Model execution and REST API |
| Containerization | Docker | Consistent application packaging |
| MLOps & DevOps | GitHub Actions | Automated validation and image builds |
| Testing & linting | PyTest, Ruff | Test execution and static code-quality checks |
| API documentation | FastAPI, Swagger UI | Interactive endpoint exploration |

---

## Performance

### Reported CPU inference latency

<escape>~71 ms</escape>

This is the reported inference latency for the optimized model execution path.

The measured value should be interpreted in the context of the hardware, image resolution, batch size, preprocessing, and postprocessing used during measurement. It should not be presented as a guaranteed latency for every request or machine.

### Recommended benchmark methodology

For a defensible performance report, measure and document:

- **Hardware:** CPU model, core configuration, memory, and operating environment.
- **Input:** Image dimensions, color format, and batch size.
- **Runtime:** ONNX Runtime version, execution provider, and thread configuration.
- **Measurement:** Warm-up iterations, number of measured runs, and timing method.
- **Scope:** Model-only inference latency versus complete API request latency.
- **Statistics:** Median, p95, and throughput where relevant.

Report both model-only latency and end-to-end API latency when possible. This separates the effect of ONNX optimization from request parsing, image preprocessing, serialization, and network overhead.

Do not claim a speedup percentage over PyTorch until both implementations have been benchmarked under equivalent conditions.

---

## API Reference

**Local base URL:** `http://localhost:8000`

**Interactive Swagger UI:** `http://localhost:8000/docs`

The endpoints below describe the intended interface. Confirm the HTTP methods, payload schema, and response format against the implemented FastAPI routes before publishing them as an exact API contract.

### `GET /health`

Checks whether the API service is reachable.

**Example request**

```bash
curl http://localhost:8000/health
```

**Illustrative response**

```json
{
  "status": "ok"
}
```

The response above is an example only. A basic health check does not necessarily verify that the model artifact is loaded or that inference is operational.

### `POST /predict`

Accepts an image and returns its predicted pedestrian segmentation.

For an implementation that accepts multipart file uploads, a request can be made using:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -F "file=@sample.jpg"
```

Replace `sample.jpg` with the path to an image supported by your application.

The request above assumes the endpoint accepts a multipart field named `file`. If the implementation uses a different field name or JSON/base64 input, adjust the request accordingly.

**Expected processing stages**

- Input validation
- Image preprocessing
- ONNX model inference
- Segmentation output interpretation
- Response serialization

Document the actual response schema, including mask dimensions, class encoding, output format, and error responses, once these are confirmed from the implementation.

### Endpoint summary

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Basic service health check |
| `POST` | `/predict` | Generate a pedestrian segmentation prediction |
| `GET` | `/docs` | Interactive Swagger UI documentation |

---

## Quickstart

### Prerequisites

Install or prepare:

- Git
- Docker Engine or Docker Desktop
- An image supported by the model for inference testing

Python is not required on the host for the Docker-based workflow, provided the image contains the application's Python runtime and dependencies.

### 1. Obtain the repository

If the project is already on your machine, navigate to its root directory.

Otherwise, clone your repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd pedestrian-segmentation-mlops
```

Replace the placeholder with the actual repository URL.

### 2. Build the Docker image

Run this command from the directory containing the `Dockerfile`:

```bash
docker build -t pedestrian-segmentation-mlops:latest .
```

The build installs the dependencies specified by the Docker configuration and packages the application into a local image.

### 3. Start the API container

```bash
docker run --rm \
  --name pedestrian-segmentation-api \
  -p 8000:8000 \
  pedestrian-segmentation-mlops:latest
```

This assumes the application listens on `0.0.0.0:8000` inside the container. Confirm the port and application startup command in your Docker configuration.

### 4. Verify service availability

In a second terminal:

```bash
curl http://localhost:8000/health
```

Then open the interactive API documentation:

```text
http://localhost:8000/docs
```

### 5. Run a prediction

If `/predict` accepts a multipart image upload using the field name `file`:

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@sample.jpg"
```

Use an image path that exists in the current terminal directory, and adapt the request to the actual endpoint schema.

### 6. Stop the container

Press `Ctrl+C` in the terminal running the container. With `--rm`, Docker removes the stopped container automatically.

### Optional: Inspect the built image

```bash
docker images pedestrian-segmentation-mlops
docker inspect pedestrian-segmentation-mlops:latest
```

These commands help verify the local image and inspect its configuration.

> **Build versus pull:** `docker build` creates an image from the local source and Dockerfile. `docker pull` downloads an image that has already been published to a registry. Since no registry image URL was provided, this quickstart builds locally rather than assuming a published image exists.

---

## CI/CD Pipeline

The repository uses GitHub Actions to automate three validation stages.

```text
Code Push / Pull Request
          |
          v
+----------------------+
| Stage 1: PyTest      |
| Automated tests      |
+----------------------+
          |
          v
+----------------------+
| Stage 2: Ruff        |
| Code-quality checks  |
+----------------------+
          |
          v
+----------------------+
| Stage 3: Docker Build|
| Image build check    |
+----------------------+
          |
          v
     Workflow Result
```

### Pipeline responsibilities

**Stage 1 — Automated testing**

PyTest executes the configured test suite to detect regressions in application behavior and inference components.

**Stage 2 — Code quality**

Ruff checks Python files against the project's configured linting rules. The exact rules depend on the repository configuration.

**Stage 3 — Container build**

Docker builds the application image to validate the container packaging process.

The workflow should be configured so that failures are visible in GitHub Actions and pull requests cannot be merged when required checks fail, if that is the intended repository policy.

### Current scope

The documented workflow covers testing, linting, and image building. Publishing images to a registry, vulnerability scanning, deployment, model monitoring, and automated rollback are separate capabilities and should only be claimed if implemented.

---

## Project Structure

The following is a recommended modular layout. Rename paths and files to match the actual repository.

```text
pedestrian-segmentation-mlops/
├── .github/
│   └── workflows/
│       └── ci.yml              # Automated quality and build checks
├── src/
│   ├── api/
│   │   ├── main.py             # FastAPI application entry point
│   │   └── routes.py           # Prediction and health routes
│   ├── inference/
│   │   ├── predictor.py        # ONNX Runtime inference
│   │   └── preprocessing.py    # Input transformations
│   └── config.py               # Application configuration
├── training/
│   ├── train.py                # Model training entry point
│   ├── model.py                # U-Net architecture
│   └── export_onnx.py          # Model export utility
├── tests/
│   ├── test_api.py             # API behavior tests
│   ├── test_inference.py       # Inference component tests
│   └── test_preprocessing.py   # Preprocessing tests
├── models/
│   └── .gitkeep                # Model artifact location
├── .dockerignore
├── .gitignore
├── Dockerfile
├── requirements.txt
├── pyproject.toml              # Ruff and tool configuration
└── README.md
```

### Separation of responsibilities

- `training/` contains model development, training, and export utilities.
- `src/inference/` handles input transformations and ONNX model execution.
- `src/api/` exposes the model through HTTP endpoints.
- `tests/` contains automated validation.
- `.github/workflows/` defines CI automation.
- `models/` represents the model-artifact location; large binary artifacts may instead be managed through a model registry or dedicated artifact storage.

This separation helps keep model development independent from serving and makes the inference path easier to test.

---

## Local Development & Validation

For developers who want to run quality checks outside the container, use the project's configured Python environment.

```bash
python -m venv .venv
```

Activate the environment:

**Windows PowerShell**

```powershell
.venv\Scripts\Activate.ps1
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

Install dependencies and run validation:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pytest
ruff check .
```

These commands assume `requirements.txt` contains the required development dependencies and that Ruff is installed in the environment. If dependencies are managed through `pyproject.toml` or another tool, use the repository's actual installation instructions.

---

## Engineering Principles

The project is organized around several practical ML engineering principles:

- **Reproducibility:** Version the code, dependencies, model configuration, and benchmark methodology.
- **Separation of concerns:** Keep training, model export, inference, and HTTP serving logically distinct.
- **Testability:** Validate preprocessing, inference behavior, endpoint responses, and error handling independently.
- **Performance measurement:** Distinguish model execution time from complete API latency.
- **Deployment consistency:** Use a container to reduce differences between development and execution environments.
- **Automated validation:** Run tests, lint checks, and container builds through a repeatable workflow.

## Roadmap

Potential next steps, if not already implemented:

- [ ] Add a reproducible PyTorch-versus-ONNX benchmark.
- [ ] Report median and p95 inference latency across repeated runs.
- [ ] Add model-artifact versioning and checksums.
- [ ] Add CI tests for invalid inputs, model loading, and inference failures.
- [ ] Publish versioned Docker images to a container registry.
- [ ] Add container vulnerability scanning.
- [ ] Add structured logs and inference timing instrumentation.
- [ ] Document segmentation metrics such as IoU and Dice score on a held-out evaluation set.
- [ ] Add an end-to-end smoke test that starts the container and exercises the prediction endpoint.

## License

Specify the repository's actual license here. Do not imply that the code, model weights, or training dataset are openly licensed unless their respective licensing terms permit it.

---

**Project focus:** Deploying a computer vision model through an optimized inference runtime, a containerized API, and automated engineering quality gates.
