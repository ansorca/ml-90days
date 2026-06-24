# ML 90 Days

A hands-on machine learning study project building core algorithms from scratch. Originally designed as a 90-day AI/ML and Computer Vision mastery roadmap (5 stages, 5 books, 20+ projects), the focus has since evolved toward the **SRE-to-AI Infrastructure** path — emphasizing Docker, Kubernetes, and alignment alongside foundational ML.

## Background

This project started from a structured 90-day plan covering:

| Stage | Days | Focus |
|-------|------|-------|
| S1 Foundations | 1–20 | Python, NumPy, linear algebra, probability, first classifiers |
| S2 Core ML | 21–45 | Trees, ensembles, SVMs, pipelines, feature engineering |
| S3 Deep Learning | 46–65 | Neural nets from scratch, PyTorch, CNNs, transformers |
| S4 Computer Vision | 66–80 | Detection, tracking, segmentation, alignment theory pivot |
| S5 Production | 81–90 | Serving, Docker, monitoring, CI/CD, bias auditing |

The current direction prioritizes the infrastructure and operations side of ML systems — containerization, orchestration, model serving, and AI safety/alignment — drawing on 20+ years of systems and SRE experience.

## What's Here

### Stage 1 — ML Classifiers (Clean Architecture)

The `stage1/` directory contains from-scratch classifier implementations organized with clean architecture layers:

```
stage1/
├── domain/classifier.py           # Classifier protocol (interface)
├── knn/
│   ├── knn_with_numpy.py          # K-Nearest Neighbors (pure NumPy)
│   ├── hybrid_knn.py              # Python/Rust hybrid KNN (PyO3)
│   └── mnist_knn.py               # KNN on MNIST
├── naive_bayes/
│   └── naive_bayes.py             # Gaussian Naive Bayes
├── adapters/
│   ├── cli/main.py                # Typer CLI
│   ├── http/serve.py              # FastAPI prediction endpoint
│   └── data/iris_loader.py        # Iris dataset loader
├── bootstrap.py                   # Classifier factory and wiring
└── test/                          # Unit tests
```

### Foundational Scripts

| File | What it covers |
|------|---------------|
| `gradient_descent.py` | Weight optimization on quadratic loss from scratch |
| `neural_layer.py` | Single-layer forward pass: matrix multiply, bias, ReLU |
| `ms_loss.py` | Mean squared loss |
| `torch_descent.py` | Gradient descent using PyTorch autograd |
| `iris.py` | Iris dataset exploration |
| `summarize_data.py` | Data summarization utilities |
| `fenner_polinomials.py` | Polynomial fitting (Fenner textbook exercises) |

### Alignment and Interpretability

| File | What it covers |
|------|---------------|
| `alignment_test.py` | Adversarial noise simulation — visualizing the gap between human perception and model vulnerability |
| `interpretability_test.py` | Sobel edge detection — what a CNN's first convolutional layer actually extracts |

These reflect the shift toward AI safety: understanding how models fail, what they "see," and where adversarial inputs exploit the gap.

## Running

### CLI

```bash
python -m stage1.adapters.cli.main --classifier knn --features "5.1,3.5,1.4,0.2"
```

Supported classifiers: `knn`, `bayes`, `hybrid`

### HTTP API

```bash
uvicorn stage1.adapters.http.serve:app
```

Set the classifier via the `CLASSIFIER` environment variable (defaults to `hybrid`).

```bash
curl -X POST http://localhost:8000/predict/ \
  -H "Content-Type: application/json" \
  -d '{"data": [[5.1, 3.5, 1.4, 0.2]]}'
```

### Tests

```bash
pytest stage1/test/
```

## Current Direction

The project continues to evolve with an SRE-to-AI infrastructure focus:

- **Containerization and orchestration** — Docker, Kubernetes for ML workloads
- **Model serving infrastructure** — FastAPI, gRPC, latency/throughput optimization
- **AI alignment** — adversarial robustness, interpretability, bias auditing
- **Rust acceleration** — performance-critical ML components via PyO3 (hybrid KNN is the first example)

## Author

Jose Antonio Soriano Camarillo
