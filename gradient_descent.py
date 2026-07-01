import numpy as np
import time

# 1. Initialization
w = 5.0            # Our initial bad weight guess
learning_rate = 0.1 # Stride length (alpha)
epochs = 40        # Number of times we repeat the optimization step

print("Starting weight value: w = {}".format(w))
print("---------------------------------------")

# 2. The Training Loop
for epoch in range(1, epochs + 1):
    # Calculate the current Loss: L = w^2
    loss = w ** 2
    
    # Calculate the Gradient (The Derivative of w^2 with respect to w is 2*w)
    # This represents the slope of the hill at our current position
    gradient = 2 * w
    
    # --- THE CHALLENGE ---
    # Apply the weight update equation to adjust 'w' in the direction of descent.
    # Formula: w_new = w_old - (learning_rate * gradient)
    # YOUR CODE HERE:
    w = w - (learning_rate * gradient)
    
    # Print progress every 5 steps
    if epoch % 5 == 0 or epoch == 1:
        print("Epoch {:02d} | Current Weight: {:8.5f} | Current Loss: {:8.5f}".format(epoch, w, loss))
        time.sleep(0.1)

print("---------------------------------------")
print("Final optimized weight value: w = {:.5f}".format(w))