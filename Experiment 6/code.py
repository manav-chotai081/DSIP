import cv2
import numpy as np 
import matplotlib.pyplot as plt

image_path = 'images.jpg'  # Replace with your image path
input_image = cv2.imread(image_path, 0)
plt.figure()
# plt.imshow(input_image)
# plt.show()

input_path = 'images.jpg'  # Replace with your image path
img = cv2.imread(input_path, 0)

kernel = np.ones((5, 5), np.uint8)

dilation = cv2.dilate(img, kernel, iterations=3)

plt.subplot(1, 2, 1)
# plt.imshow(dilation, cmap='gray')

erosion = cv2.erode(img, kernel, iterations=3)

plt.subplot(1, 2, 2)
plt.imshow(erosion, cmap='gray')

# kernel = np.ones((5, 5), np.uint8)
# plt.figure()
# plt.subplot(1, 2, 1)
# plt.imshow(input_image, cmap='gray')
# plt.title('Original')
# plt.subplot(1, 2, 2)
# dilation = cv2.dilate(input_image, kernel, iterations=3)
# plt.subplot(1, 2, 1)
# plt.imshow(dilation, cmap='gray')