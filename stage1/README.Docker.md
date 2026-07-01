# ml-90days — Docker Guide

A FastAPI service exposing ML classifiers (KNN, Naive Bayes, hybrid Rust/Python KNN).
The Rust extension is compiled inside Docker via a multi-stage build — no pre-built binaries required.

## Running with Docker Compose

```bash
# From stage1/
docker compose up --build
```

The service will be available at http://localhost:8000.

First build is slow — Rust compilation happens inside the `rust-builder` stage.
Subsequent builds use the Docker layer cache unless `rust_knn/` source changes.

## Running from Docker Hub

```bash
docker run -p 8000:8000 ansorca/ml-90days:latest
```

## Selecting a classifier

The classifier is controlled by the `CLASSIFIER` environment variable.
Accepted values: `hybrid` (default), `knn`, `bayes`.

```bash
docker run -p 8000:8000 -e CLASSIFIER=bayes ansorca/ml-90days:latest
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/predict/` | Run inference. Body: `{"data": [[f1, f2, f3, f4]]}` |
| GET | `/health/live` | Liveness — is the process running? |
| GET | `/health/ready` | Readiness — is the model loaded and ready? |

Example prediction request:

```bash
curl -X POST http://localhost:8000/predict/ \
  -H "Content-Type: application/json" \
  -d '{"data": [[5.1, 3.5, 1.4, 0.2]]}'
```

## Image architecture

The Dockerfile uses a two-stage build:

1. **rust-builder** — installs Rust + maturin, compiles `rust_knn` into a Python wheel
2. **base** — installs Python dependencies, copies the wheel from stage 1, runs the app

This keeps the final image free of the Rust toolchain (~1GB saved).
