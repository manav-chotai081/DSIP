import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image in grayscale
src_image = cv2.imread("ex1_1.png", cv2.IMREAD_GRAYSCALE)

if src_image is None:
    print("Image not found!")
    exit()

# -------------------------
# Contrast Stretching
# -------------------------
r_min = np.min(src_image)
r_max = np.max(src_image)

stretched_image = (src_image - r_min) / (r_max - r_min) * 255
stretched_image = np.uint8(stretched_image)

# -------------------------
# Display Images
# -------------------------
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(src_image, cmap='gray')
plt.title("Original")
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(stretched_image, cmap='gray')
plt.title(f"Contrast Stretched (rmin={r_min}, rmax={r_max})")
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
plt.hist(stretched_image.ravel(), bins=256, range=(0,255))
plt.title("Histogram After")

plt.show()