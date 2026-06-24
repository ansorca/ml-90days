import timeit
import numpy as np
from stage1.knn.hybrid_knn import HybridKNNClassifier
from stage1.knn.knn_with_numpy import KNNClassifier


def test_hybrid_knn_execution_time():
    np.random.seed(42)
    num_features = 10
    samples_scale = 50000

    k_value = 3
    repeat_times = 3
    timer_argument=100

    X_large = np.random.randn(samples_scale, num_features).astype(np.float32)
    y_large = np.random.randint(0, 2, size=samples_scale).astype(np.int64)
    x_query = np.random.randn(1, num_features).astype(np.float32)

    knn_hybrid = HybridKNNClassifier(k=k_value)
    knn_python =KNNClassifier(k=k_value)

    knn_hybrid.fit(X_large, y_large)
    knn_python.fit(X_large, y_large)

    time_hybrid = min(timeit.Timer(lambda: knn_hybrid.predict(x_query)).repeat(repeat=repeat_times, number=timer_argument))
    time_python = min(timeit.Timer(lambda: knn_python.predict(x_query)).repeat(repeat=repeat_times, number=timer_argument))
    print("Large Dataset (50,000 samples) 100 Loops Hybrid-Rust Time: {:.5f} seconds".format(time_hybrid))
    print("Large Dataset (50,000 samples) 100 Loops NumPy Time: {:.5f} seconds".format(time_python))
    speedup = time_python/time_hybrid
    assert 1.5 <=speedup, f"Expected 1.5x speedup, got {speedup:.2f}x"

