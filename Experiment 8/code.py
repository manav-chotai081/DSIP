import cv2
import matplotlib.pyplot as plt

# Load the image
image_path = "D:\\ict\\sem5\\DSIP\\lab & theory session\\assignmentAndOEQ\\Experiment 8\\o.jpg"   # Replace with your image path
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if image is loaded
if image is None:
    print("Image not found!")
    exit()

# -------------------------
# Histogram Before Equalization
# -------------------------
histogram = cv2.calcHist([image], [0], None, [256], [0, 256])

plt.figure(figsize=(8, 6))
plt.title("Histogram Before Equalization")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.plot(histogram)
plt.xlim([0, 256])
plt.grid(True)
plt.show()

# -------------------------
# Perform Histogram Equalization
# -------------------------
equalized_image = cv2.equalizeHist(image)

# -------------------------
# Display Original and Equalized Images
# -------------------------
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Equalized Image")
plt.imshow(equalized_image, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()

# -------------------------
# Histogram After Equalization
# -------------------------
equalized_histogram = cv2.calcHist([equalized_image], [0], None, [256], [0, 256])

plt.figure(figsize=(8, 6))
plt.title("Histogram After Equalization")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.plot(equalized_histogram)
plt.xlim([0, 256])
plt.grid(True)
plt.show()