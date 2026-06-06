import knn_with_numpy
import timeit
from memory_profiler import profile
import numpy as np

# --- PROFILING ENVIRONMENT ---

# Let's generate a mock classification dataset
np.random.seed(42)
num_features = 10

# Small Dataset (1,000 reference samples)
X_small = np.random.randn(1000, num_features)
y_small = np.random.randint(0, 2, size=1000)

# Large Dataset (50,000 reference samples) - Simulating production scale
X_large = np.random.randn(50000, num_features)
y_large = np.random.randint(0, 2, size=50000)

# A single test input vector we want to classify
x_test_input = np.random.randn(1, num_features)

# Instantiate our models
knn_small = knn_with_numpy.KNNClassifier(k=3)
knn_small.fit(X_small, y_small)

knn_large = knn_with_numpy.KNNClassifier(k=3)
knn_large.fit(X_large, y_large)


# --- TIME COMPLEXITY PROFILE ---
print("--- TIME PROFILE CHECKS ---")

# Measure execution time on the small dataset
time_small = min(timeit.Timer(lambda: knn_small.predict(x_test_input)).repeat(repeat=3, number=100))
print("Small Dataset (1,000 samples) 100 loops time:  {:.5f} seconds".format(time_small))

# Measure execution time on the large dataset
time_large = min(timeit.Timer(lambda: knn_large.predict(x_test_input)).repeat(repeat=3, number=100))
print("Large Dataset (50,000 samples) 100 loops time: {:.5f} seconds".format(time_large))


# --- SPACE COMPLEXITY PROFILE ---
# We decorate our execution function to track memory spikes in real time
@profile(precision=4)
def run_memory_audit():
    print("\n--- SPACE PROFILE AUDIT ---")
    # Triggering the large dataset prediction to monitor VRAM/RAM footprints
    _ = knn_large.predict(x_test_input)

if __name__ == "__main__":
    run_memory_audit()