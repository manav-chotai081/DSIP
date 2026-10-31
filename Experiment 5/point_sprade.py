import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread(r'D:\\ict\\sem5\\DSIP\\lab & theory session\\assignmentAndOEQ\\Experiment 6\\Screenshot 2026-07-27 201259.png')

img = img.astype(np.float32) / 255.0

size = 15

center = size // 2
x = np.arange(-7, 8)

X, Y = np.meshgrid(x, x)

sigma = 3

gaussian_psf = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
gaussian_psf = gaussian_psf / np.sum(gaussian_psf)
# print('Gaussian PSF:', gaussian_psf)

horizontal_psf = np.zeros((size,size), dtype = np.float32)
horizontal_psf[center, :] = 1
horizontal_psf = horizontal_psf / np.sum(horizontal_psf)


vertical_psf = np.zeros((size,size), dtype = np.float32)
vertical_psf[center, :] = 1
vertical_psf = vertical_psf / np.sum(vertical_psf)

diagonal_psf = np.zeros((size,size), dtype = np.float32)
np.fill_diagonal(diagonal_psf, 1)
diagonal_psf = diagonal_psf / np.sum(diagonal_psf)

circular_psf = np.zeros((size,size), dtype = np.float32)
Yc, Xc = np.ogrid[:size, :size]
distance = np.sqrt((Xc-center)**2 + (Yc-center)**2)
circular_psf[distance <= 6] = 1
circular_psf = circular_psf / np.sum(circular_psf)

ring_psf = np.zeros((size,size), dtype = np.float32)
ring_psf[(distance >= 4) & (distance <= 6)] = 1
ring_psf = ring_psf / np.sum(ring_psf)
