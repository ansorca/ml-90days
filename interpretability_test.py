import cv2
import numpy as np

# 1. Create our familiar 400x400 canvas with a white square in the center
image = np.zeros((400, 400), dtype=np.uint8) # Grayscale for simpler math
image[150:250, 150:250] = 255

# 2. Define a Sobel Horizontal Edge Filter (A 3x3 mathematical matrix)
# This matrix acts like a lens that amplifies vertical transitions (left/right edges)
sobel_horizontal = np.array([[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]])

# 3. Apply the filter using a 16-bit signed integer to keep negative numbers
# original step 3 code: filtered_image = cv2.filter2D(image, -1, sobel_horizontal)
filtered_16bit = cv2.filter2D(image, cv2.CV_16S, sobel_horizontal)

# Take the absolute value to turn negative edges into positive brightness
filtered_absolute = np.absolute(filtered_16bit)

# Convert back to standard 8-bit image format for displaying
filtered_image = np.uint8(filtered_absolute)

# 4. Show the original structural input vs. what the "early layer" actually extracts
display = np.hstack((image, filtered_image))
print("Press any key on the window to exit.")
cv2.imshow("Left: Human Input | Right: What the First Layer Sees (Edges)", display)
cv2.waitKey(0)
cv2.destroyAllWindows()
