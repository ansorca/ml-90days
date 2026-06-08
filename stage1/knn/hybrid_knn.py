import numpy as np
import timeit
# Import your newly compiled native Rust function directly!
import rust_knn 

class HybridKNNClassifier:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        # Store as continuous float32 vectors to match Rust types perfectly
        self.X_train = np.ascontiguousarray(X, dtype=np.float32)
        self.y_train = np.ascontiguousarray(y, dtype=np.int64)

    def predict_single(self, x_test):
        # Ensure the test vector matches continuous memory requirements
        x_test_clean = np.ascontiguousarray(x_test.flatten(), dtype=np.float32)
        
        # OFF-LOAD TO RUST: Eliminate the python interpreter overhead
        distances = rust_knn.compute_distances(x_test_clean, self.X_train)
        
        # Keep native sorting structures in NumPy for convenience
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices]
        
        return np.argmax(np.bincount(k_nearest_labels))

    def predict(self, X_test):
        return np.array([self.predict_single(x) for x in X_test])


# --- VERIFYING EXECUTION TIME GAINS ---
np.random.seed(42)
num_features = 10
samples_scale = 50000

X_large = np.random.randn(samples_scale, num_features).astype(np.float32)
y_large = np.random.randint(0, 2, size=samples_scale).astype(np.int64)
x_query = np.random.randn(1, num_features).astype(np.float32)

# Fit Hybrid Model
knn_hybrid = HybridKNNClassifier(k=3)
knn_hybrid.fit(X_large, y_large)

# Time 100 loops of the hybrid execution path
time_hybrid = min(timeit.Timer(lambda: knn_hybrid.predict(x_query)).repeat(repeat=3, number=100))
print("Large Dataset (50,000 samples) 100 Loops Hybrid-Rust Time: {:.5f} seconds".format(time_hybrid))