import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

# Original 5x5 image
image = np.array([
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [1, 1, 1, 1, 1],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0]
])

# Horizontal edge detector
horizontal_filter = np.array([
    [-1, -1, -1],
    [ 0,  0,  0],
    [ 1,  1,  1]
])

# Apply convolution
result = convolve2d(image, horizontal_filter, mode='same', boundary='fill', fillvalue=0)

# Plot side-by-side
plt.figure()

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.colorbar()
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Horizontal Edge Detection")
plt.imshow(result, cmap='gray')
plt.colorbar()
plt.axis('off')

# Save the figure
plt.savefig("edge_detection.png", dpi=300, bbox_inches='tight')

# Show the figure
plt.show()