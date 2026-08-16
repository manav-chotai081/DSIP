import cv2
import matplotlib.pyplot as plt

src_image = cv2.imread("ex1_2.png", cv2.IMREAD_GRAYSCALE)

if src_image is None:
    print("Image not found!")
    exit()

# -------------------------
# Histogram Equalization
# -------------------------
equalized_image = cv2.equalizeHist(src_image)

# -------------------------
# Display Images
# -------------------------
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(src_image, cmap='gray')
plt.title("Original")
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(equalized_image, cmap='gray')
plt.title("Histogram Equalized")
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
plt.hist(equalized_image.ravel(), bins=256, range=(0,255))
plt.title("Histogram After")

plt.show()