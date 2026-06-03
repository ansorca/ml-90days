import cv2
import numpy as np

# 1. Recreate our base image: A perfect 400x400 black canvas with a white square
clean_image = np.zeros((400, 400, 3), dtype=np.uint8)
clean_image[150:251, 150:251] = [255, 255, 255]

# 2. Generate "Adversarial Noise"
# Real adversarial attacks generate specific mathematical patterns, 
# but we will simulate it using highly controlled, subtle pixel fluctuations.
np.random.seed(42) # Ensuring consistency
noise = np.random.randint(-5, 6, (400, 400, 3), dtype=np.int8)

# 3. Apply the noise to the clean image
# We convert to int16 temporarily to prevent numbers from clipping outside 0-255 bounds
adversarial_matrix = clean_image.astype(np.int16) + noise
adversarial_matrix = np.clip(adversarial_matrix, 0, 255).astype(np.uint8)

# 4. Calculate the mathematical difference
# To a human, this looks like black space, but to a model, this is the entire "exploit"
pixel_diff = cv2.absdiff(clean_image, adversarial_matrix)

# 5. Stack them horizontally to compare: Clean vs. Adversarial vs. Exploit Pattern
display_canvas = np.hstack((clean_image, adversarial_matrix, pixel_diff * 20)) # Multiplied diff by 20 so humans can see it

print("Press any key on the image window to close.")
cv2.imshow("Left: Clean | Center: Adversarial (looks identical) | Right: Amplified Exploit", display_canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()