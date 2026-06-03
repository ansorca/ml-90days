import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss


def summarize(data):
    result = {"count": len(data)}
    total = 0.0
    for d in data:
        total += d
        if result.get("min", None) is None or d < result["min"]:
            result["min"] = d
        if result.get("max", None) is None or d > result["max"]:
            result["max"] = d
    result["mean"] = round(total / result["count"], 2)
    result["max"] = round(result["max"], 2)
    result["min"] = round(result["min"], 2)
    return result


data = [10, 20, 30, 40, 50]
summarize(data)


def gemini_challenge():
    # 1. Create a fake black "image" (400x400 pixels, 3 color channels)
    image = np.zeros((400, 400, 3), dtype=np.uint8)

    # 2. Draw a solid white square right in the center (from pixel 150 to 250)
    # HINT: In NumPy, you slice arrays like this: image[y_start:y_end, x_start:x_end] = [B, G, R]
    # YOUR CODE HERE: Make the center square white [255, 255, 255]
    image[150:251, 150:251] = [255, 255, 255]
    # 3. Change the color of just the top-left 50x50 corner to bright Green
    # YOUR CODE HERE: [0, 255, 0]
    image[0:51, 0:51] = [0, 255, 0]

    # 4. Display the image in a window
    cv2.imshow("My First Matrix Image", image)
    cv2.waitKey(0)  # Waits until you press any key to close
    cv2.destroyAllWindows()


gemini_challenge()
