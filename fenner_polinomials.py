import matplotlib.pyplot as plt
import numpy as np


fig, axes = plt.subplots(2, 2)
fig.tight_layout()
titles = [
    "$y=c_0$",
    "$y=c_1x+c_0$",
    "$y=c_2x^2+c_1x+c_0$",
    "$y=c_3x^3+c_2x^2+c_1x+c_0$",
]
xs = np.linspace(-10, 10, 100)

for power, (ax, title) in enumerate(zip(axes.flat, titles), 1):
    coeffs = np.random.uniform(-5, 5, power)
    poly = np.poly1d(coeffs)
    ax.plot(xs, poly(xs))
    ax.set_title(title)
