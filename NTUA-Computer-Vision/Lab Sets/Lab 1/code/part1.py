
from cv2 import imread
import numpy as np
import utils
import EdgeDetect
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pdb

THETA_REAL = 1
THETA_NOISY1L = 0.1
THETA_NOISY1M = 0.1
SIGMA_1 = 1.5
PSNR1 = 20

THETA_NOISY2L = 0.2
THETA_NOISY2M = 0.2
THETA_VENICE = 80
SIGMA_2 = 3
PSNR2 = 10

np.set_printoptions(precision=3)

# Step 1.1.1: Open images
X = imread('images/edgetest_10.png', 0)

# Step 1.1.2 Add noise to images
X_n1, sigma_1 = utils.imnoise(X, PSNR=PSNR1)
X_n2, sigma_2 = utils.imnoise(X, PSNR=PSNR2)

fig = plt.figure(1)
fig.suptitle('Images with and without noise')
ax = plt.subplot(131)
plt.imshow(X, cmap=plt.get_cmap('gray'))
ax.set_title('Original')

ax = plt.subplot(132)
plt.imshow(X_n1, cmap=plt.get_cmap('gray'))
ax.set_title('PSNR = 20dB')

ax = plt.subplot(133)
plt.imshow(X_n2, cmap=plt.get_cmap('gray'))
ax.set_title('PSNR = 10dB')

''' Step 1.2.1 - Create Filters '''


''' Perform Edge Detection '''
fig = plt.figure(2)
fig.suptitle('Edge detection')


''' Non noisy image '''
ax = plt.subplot(231)
plt.imshow(X, cmap=plt.get_cmap('gray'))
ax.set_title('Original Image')

T = utils.boundary_operator(X, utils.MorphologicalFilters.FIVE_POINT_FILTER) > THETA_REAL
ax = plt.subplot(232)
plt.imshow(T, cmap=plt.get_cmap('gray'))
ax.set_title('Original Image Edges')


''' Noisy images '''
# Noisy image 1
nimgL1 = EdgeDetect.EdgeDetect(X_n1, SIGMA_1, THETA_NOISY1L, 'linear')
nimgM1 = EdgeDetect.EdgeDetect(X_n1, SIGMA_1, THETA_NOISY1M, 'morph')

nimgL2 = EdgeDetect.EdgeDetect(X_n2, SIGMA_2, THETA_NOISY2L, 'linear')
nimgM2 = EdgeDetect.EdgeDetect(X_n2, SIGMA_2, THETA_NOISY2M, 'morph')

ax = plt.subplot(233)
plt.imshow(nimgL1, cmap=plt.get_cmap('gray'))
ax.set_title('Noisy image with PSNR 20dB, linear filter.')


ax = plt.subplot(234)
plt.imshow(nimgM1, cmap=plt.get_cmap('gray'))
ax.set_title('Noisy image with PSNR 20dB, morphological filter.')


ax = plt.subplot(235)
plt.imshow(nimgL2, cmap=plt.get_cmap('gray'))
ax.set_title('Noisy image with PSNR 10dB, linear filter.')


ax = plt.subplot(236)
plt.imshow(nimgM2, cmap=plt.get_cmap('gray'))
ax.set_title('Noisy image with PSNR 10dB, morphological filter')

''' Evaluate Quality of Edge Detector '''
print('Noisy image with PSNR 20dB, linear filter. C = {}'.format(utils.quality_factor(nimgL1, T)))
print('Noisy image with PSNR 20dB, morphological filter. C = {}'.format(utils.quality_factor(nimgM1, T)))
print('Noisy image with PSNR 10dB, linear filter. C = {}'.format(utils.quality_factor(nimgL2, T)))
print('Noisy image with PSNR 10dB, morphological filter. C = {}'.format(utils.quality_factor(nimgM2, T)))

''' Step 1.4.1 - Test on real image'''
venice = imread('images/urban_edges.jpg', 0)
fig = plt.figure(3)
fig.suptitle('Edge detection on real images')
ax = plt.subplot(121)
plt.imshow(venice, cmap=plt.get_cmap('gray'))

ax = plt.subplot(122)
binary_venice = utils.boundary_operator(venice, utils.MorphologicalFilters.FIVE_POINT_FILTER) > THETA_VENICE
plt.imshow(binary_venice, cmap=plt.get_cmap('gray'))

plt.show()
