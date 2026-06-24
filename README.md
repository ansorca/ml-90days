# ML 90 Days

A hands-on machine learning study project implementing core algorithms from scratch using Python and NumPy.

## Project Structure

```
ml-90days/
├── stage1/                        # Clean architecture ML classifiers
│   ├── domain/classifier.py       # Classifier protocol (interface)
│   ├── knn/
│   │   ├── knn_with_numpy.py      # K-Nearest Neighbors (pure NumPy)
│   │   ├── hybrid_knn.py          # KNN with Rust-accelerated distance computation
│   │   └── mnist_knn.py           # KNN applied to MNIST
│   ├── naive_bayes/
│   │   └── naive_bayes.py         # Gaussian Naive Bayes classifier
│   ├── adapters/
│   │   ├── cli/main.py            # Typer CLI for running predictions
│   │   ├── http/serve.py          # FastAPI prediction endpoint
│   │   └── data/iris_loader.py    # Iris dataset loader
│   ├── bootstrap.py               # Classifier factory and wiring
│   └── test/                      # Tests for KNN and Naive Bayes
├── gradient_descent.py            # Gradient descent from scratch
├── neural_layer.py                # Forward pass of a single neural layer
├── ms_loss.py                     # Mean squared loss
├── torch_descent.py               # Gradient descent with PyTorch
├── iris.py                        # Iris dataset exploration
├── summarize_data.py              # Data summarization utilities
├── fenner_polinomials.py          # Polynomial fitting
├── alignment_test.py              # Model alignment experiments
├── interpretability_test.py       # Model interpretability experiments
└── word_frequencies.rb            # Word frequency counter (Ruby)
```

## Algorithms Implemented

- **K-Nearest Neighbors** - Euclidean distance, majority vote classification, configurable k
- **Hybrid KNN** - Python/Rust hybrid using PyO3 for accelerated distance computation
- **Gaussian Naive Bayes** - Prior probabilities, Gaussian likelihood per feature
- **Gradient Descent** - Weight optimization on a simple quadratic loss
- **Neural Layer Forward Pass** - Matrix multiplication, bias addition, ReLU activation

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
