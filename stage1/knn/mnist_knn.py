from sklearn.datasets import fetch_openml
import numpy as np
from knn_with_numpy import KNNClassifier

mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(int)

# Normalize pixels to 0-1
X = X / 255.0

# Use a small subset first — your k-NN will be slow on full dataset

# Results: 1k training = 83% @ k=1, ~fast
#          10k training = 97% @ k=1, ~slow
# Demonstrates data quantity vs compute tradeoff


X_train, X_test = X[:1000], X[1000:1100]
y_train, y_test = y[:1000], y[1000:1100]
k = 4

knn = KNNClassifier(1)
knn.fit(X_train, y_train)

predictions = knn.predict(X_test)
accuracy = np.sum(np.array(predictions) == y_test) / len(y_test)
print(f"K= {k} Training examples: {len(X_train)} Accuracy: {accuracy:.2%}")


X_train, X_test = X[:10000], X[10000:10100]
y_train, y_test = y[:10000], y[10000:10100]
knn = KNNClassifier(1)
knn.fit(X_train, y_train)

predictions = knn.predict(X_test)
accuracy = np.sum(np.array(predictions) == y_test) / len(y_test)
print(f"K= {k} Training examples: {len(X_train)} Accuracy: {accuracy:.2%}")