import numpy as np

# 1. Input Data (X): 4 pixel values from an image patch
# Matrix shape: (1, 4) -> 1 sample, 4 features
X = np.array([[50, 120, 230, 10]], dtype=np.float32)

# 2. Weights Matrix (W): Randomly initialized for 3 neurons
# Each neuron must have exactly 4 weights (one for each input)
# Matrix shape: (4, 3) -> 4 inputs, 3 neurons
np.random.seed(42) # Keeps random numbers identical for testing
W = np.random.randn(4, 3).astype(np.float32)

# 3. Bias Vector (b): One offset tuning value per neuron
# Vector shape: (1, 3) -> 1 bias for each of the 3 neurons
b = np.array([[2.0, -1.0, 0.5]], dtype=np.float32)

# --- THE CHALLENGE ---
# Using NumPy's matrix multiplication function, compute the layer's output (Z).
# Hint: In NumPy, matrix multiplication is performed using np.dot(A, B) or the @ operator (A @ B)
# Formula: Z = XW + b

Z = np.dot(X, W) + b

# --- PRINTING THE RESULTS ---
print("--- Input Matrix X (Shape: {}) ---".format(X.shape))
print(X)
print("\n--- Weights Matrix W (Shape: {}) ---".format(W.shape))
print(W)
print("\n--- Bias Vector b (Shape: {}) ---".format(b.shape))
print(b)
print("\n--- Computed Layer Output Z (Shape: {}) ---".format(Z.shape if Z is not None else "None"))
print(Z)

# Apply the Rectified Linear Unit (ReLU) activation function
# np.maximum compares every element in Z against 0 and keeps the higher value
A = np.maximum(0, Z)

print("\n--- Activated Layer Output A (Shape: {}) ---".format(A.shape))
print(A)