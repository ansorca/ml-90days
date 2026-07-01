# ml-90days

A learning project for Python, Docker, Kubernetes, and CI/CD.
The end goal is to build skills for an SRE/DevOps role, and eventually AI infrastructure.

## Project structure

```
ml-90days/
  stage1/               # Python package: ML classifiers exposed via FastAPI
    adapters/
      http/             # FastAPI app (serve.py, schemas.py)
      cli/              # CLI entrypoint
      data/             # Data loaders (Iris dataset)
    domain/             # Abstract base classes (Classifier)
    knn/                # KNN implementations (pure Python, hybrid Rust/Python)
    naive_bayes/        # Naive Bayes implementation
    rust_knn/           # Rust extension (PyO3 + maturin) — compiled inside Docker
    bootstrap.py        # Wires together classifiers and data loaders
    test/               # pytest tests
```

## Running locally (development)

```bash
# From ml-90days/
uvicorn stage1.adapters.http.serve:app --reload
```

## Running tests

```bash
# From ml-90days/
pytest
```

## Running with Docker

```bash
# From stage1/
docker compose up --build
```

First build is slow — Rust compilation happens inside Docker via a multi-stage build.
Subsequent builds are cached unless rust_knn/ source changes.

