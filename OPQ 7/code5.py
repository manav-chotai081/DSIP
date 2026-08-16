import cv2
import numpy as np
import matplotlib.pyplot as plt

src_image = cv2.imread("ex1_5.png", cv2.IMREAD_GRAYSCALE)

if src_image is None:
    print("Image not found!")
    exit()

# -------------------------
# Log Transformation
# -------------------------
src_float = src_image.astype(np.float64)
c = 255 / np.log(1 + np.max(src_float))

log_image = c * np.log(1 + src_float)
log_image = np.uint8(np.clip(log_image, 0, 255))

# -------------------------
# Display Images
# -------------------------
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(src_image, cmap='gray')
plt.title("Original")
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(log_image, cmap='gray')
plt.title("Log Transformed")
plt.axis('off')

plt.show()

# -------------------------
# Histograms
# -------------------------
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.hist(src_image.ravel(), bins=256, range=(0,255))
plt.title("Histogram Before")

plt.subplot(1,2,2)
plt.hist(log_image.ravel(), bins=256, range=(0,255))
plt.title("Histogram After")

plt.show()