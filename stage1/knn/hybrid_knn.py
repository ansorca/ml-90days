import numpy as np
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
