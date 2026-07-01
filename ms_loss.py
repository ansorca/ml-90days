import numpy as np

# 1. Ground Truth (Y): The actual, true X-coordinates of the object across 5 frames
Y_true = np.array([120.0, 155.0, 200.0, 85.0, 310.0], dtype=np.float32)

# 2. Model Predictions (Y_hat): The coordinates predicted by your neural network
Y_pred = np.array([122.0, 140.0, 205.0, 85.0, 280.0], dtype=np.float32)

# --- THE CHALLENGE ---
# Calculate the Mean Squared Error (MSE) using pure NumPy.
# Do NOT use any standard python loops or list comprehensions.
#
# Hints for the NumPy sequence:
# - Subtraction between two arrays of the same shape happens element-wise: A - B
# - Squaring an entire array can be done with the exponent operator: A ** 2
# - You can find the mean average of a NumPy array using the function: np.mean(A)

# Calculate the final single scalar value for the loss
differences = (Y_true - Y_pred)
mse_loss = np.dot(differences, differences) / len(Y_true)

# another way
squared_differences = differences ** 2
mse_way2 = np.mean(squared_differences)
# --- EVALUATION ---
print("Actual Coordinates:   ", Y_true)
print("Predicted Coordinates:", Y_pred)
print("\nCalculated MSE Loss: ", mse_loss, mse_way2)