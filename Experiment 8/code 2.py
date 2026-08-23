import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Load Source and Reference Images
# -------------------------
source_path = "D:\\ict\\sem5\\DSIP\\lab & theory session\\assignmentAndOEQ\\Experiment 8\\o.jpg"      # Source image
reference_path = "D:\\ict\\sem5\\DSIP\\lab & theory session\\assignmentAndOEQ\\Experiment 8\\r.png"            # Reference image

source_image = cv2.imread(source_path, cv2.IMREAD_GRAYSCALE)
reference_image = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)

# Check if images are loaded
if source_image is None or reference_image is None:
    print("Source or Reference image not found!")
    exit()

# -------------------------
# Calculate Histograms
# -------------------------
source_hist = cv2.calcHist([source_image], [0], None, [256], [0, 256])
reference_hist = cv2.calcHist([reference_image], [0], None, [256], [0, 256])

# Normalize Histograms
source_hist /= source_hist.sum()
reference_hist /= reference_hist.sum()

# -------------------------
# Calculate Cumulative Distribution Functions (CDF)
# -------------------------
source_cdf = source_hist.cumsum()
reference_cdf = reference_hist.cumsum()

# -------------------------
# Perform Histogram Matching
# -------------------------
mapping = np.interp(source_cdf, reference_cdf, range(256))
matched_image = mapping[source_image].astype(np.uint8)

# -------------------------
# Display Images
# -------------------------
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.title("Source Image")
plt.imshow(source_image, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Reference Image")
plt.imshow(reference_image, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Matched Image")
plt.imshow(matched_image, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()

# -------------------------
# Display Histograms
# -------------------------
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("Source Histogram")
plt.plot(source_hist)

plt.subplot(1, 3, 2)
plt.title("Reference Histogram")
plt.plot(reference_hist)

plt.subplot(1, 3, 3)
plt.title("Matched Histogram")
matched_hist = cv2.calcHist([matched_image], [0], None, [256], [0, 256])
plt.plot(matched_hist)

plt.tight_layout()
plt.show()