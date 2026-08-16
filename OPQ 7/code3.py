import cv2
import numpy as np
import matplotlib.pyplot as plt

src_image = cv2.imread("ex1_3.png", cv2.IMREAD_GRAYSCALE)

if src_image is None:
    print("Image not found!")
    exit()

# -------------------------
# Gamma (Power-Law) Transformation
# -------------------------
gamma = 0.5  # < 1 brightens dark images

normalized_image = src_image / 255.0
gamma_corrected_image = np.power(normalized_image, gamma)
gamma_corrected_image = np.uint8(gamma_corrected_image * 255)

# -------------------------
# Display Images
# -------------------------
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(src_image, cmap='gray')
plt.title("Original")
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(gamma_corrected_image, cmap='gray')
plt.title(f"Gamma Corrected (γ={gamma})")
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
plt.hist(gamma_corrected_image.ravel(), bins=256, range=(0,255))
plt.title("Histogram After")

plt.show()