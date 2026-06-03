import torch
import time

# 1. Initialize our weight parameter as a PyTorch Tensor
# CRITICAL: We pass 'requires_grad=True' to tell PyTorch to track the calculus history 
# of this variable so it can automatically calculate the gradient later.
w = torch.tensor([5.0], requires_grad=True)
learning_rate = 0.1
epochs = 40

print("Starting weight value: w = {}".format(w.item()))
print("---------------------------------------")

for epoch in range(1, epochs + 1):
    # Forward Pass: Define the Loss function L = w^2
    loss = w ** 2
    
    # Backward Pass: The Autograd Engine calculates the gradients automatically!
    # This single line traverses backward through the mathematical operations
    loss.backward()
    
    # Update Weights using Gradient Descent
    # CRITICAL ALIGNMENT GOTCHA: We must wrap the weight modification inside 
    # 'torch.no_grad()' so PyTorch doesn't accidentally track our adjustment math 
    # as part of the model's actual architecture.
    with torch.no_grad():
        w -= learning_rate * w.grad
        
    # Manually zero out the gradients after the update step.
    # PyTorch accumulates gradients by default; if we don't clear them, 
    # the next step's math will be completely corrupted!
    w.grad.zero_()
    
    # Print progress every 5 steps
    if epoch % 5 == 0 or epoch == 1:
        print("Epoch {:02d} | Current Weight: {:8.5f} | Current Loss: {:8.5f}".format(epoch, w.item(), loss.item()))
        time.sleep(0.1)

print("---------------------------------------")
print("Final optimized weight value: w = {:.5f}".format(w.item()))