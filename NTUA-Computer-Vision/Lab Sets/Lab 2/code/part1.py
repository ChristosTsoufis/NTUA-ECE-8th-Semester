# COMPUTER VISION
# LAB PROJECT #2
# SUBJECT: Optical Flow Estimation & Feature Extraction in Videos for Movement Recognition


##########################################################################################
# Libraries & necessary packets
##########################################################################################


import numpy as np
import importlib
import math
import cv2
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import scipy.io
import skimage
from scipy.stats import multivariate_normal
from scipy.ndimage import label, generate_binary_structure, map_coordinates

import warnings
warnings.filterwarnings('ignore')


##########################################################################################
# Part 1: Tracking of Face and Hands Using Optical Flow Method of Lucas-Kanade
##########################################################################################

##########################################################################################
# Part 1.1: Skin Detection of Face and Hands 
##########################################################################################

##########################################################################################
# Reading of Skin Samples in RGB Format from the given file "skinSamplesRGB.mat"

skin_samples_RGB = scipy.io.loadmat('skinSamplesRGB.mat')['skinSamplesRGB']

##########################################################################################
# Function for Gaussian Probability Density

def Gaussian(skin_samples_RGB):
    
    skin_samples_YCbCr = cv2.cvtColor(skin_samples_RGB, cv2.COLOR_RGB2YCrCb)
    skin_samples_YCbCr.reshape((-1, 3))
    skin_samples_YCbCr = np.vstack(skin_samples_YCbCr)
    
    skin_samples_CbCr = skin_samples_YCbCr[:, 1:]
    # mean
    mean = np.mean(skin_samples_CbCr, axis = 0)
    # covariance
    covariance = np.cov(skin_samples_CbCr[:, 0], skin_samples_CbCr[:, 1])

    return mean, covariance

# Print mean & covariance

mean_CbCr, covariance_CbCr = Gaussian(skin_samples_RGB)
print("Mean vector: ")
print(mean_CbCr)
print("Covariance matrix: ")
print(covariance_CbCr)
print()


##########################################################################################
# Gaussian Probability Distribution

x_axis, y_axis = np.mgrid[140:175:1, 90:120:1]
rv = multivariate_normal(mean_CbCr, covariance_CbCr)

rcParams['figure.figsize'] = [8, 6]
position = np.dstack((x_axis, y_axis))
plt.contourf(x_axis, y_axis, rv.pdf(position))
plt.title('Gaussian Probability Density Function (in CbCr space)')
plt.ylabel('Cb Channel')
plt.xlabel('Cr Channel')

plt.show()


##########################################################################################
# Gaussian Probability Distribution in 3D

fig = plt.figure(figsize=(25, 10))

ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

ax.plot_wireframe(x_axis, y_axis, rv.pdf(position)/np.max(rv.pdf(position)), color='blue')
ax.set_zlabel('Normalized Probability')
plt.title('Gaussian Probability Distribution (in 3D space)', y = 1)
plt.ylabel('Cb Channel')
plt.xlabel('Cr Channel')

plt.show()


##########################################################################################
# Application for the 1st Image

image1_source = 'GreekSignLanguage/1.png'
image1 = np.array(cv2.imread(image1_source, 1))
image1_YCbCr = cv2.cvtColor(image1, cv2.COLOR_BGR2YCR_CB)

cb_channel = image1_YCbCr[:, :, 1]
cr_channel = image1_YCbCr[:, :, 2]

X = np.array([cb_channel, cr_channel])
positions = np.dstack((cb_channel, cr_channel))
pdf = rv.pdf(positions)/np.max(rv.pdf(positions))

# threshold = 0.1
threshold = 0.2

binary_skin = (pdf>threshold)*1
binary_skin = binary_skin.astype('uint8')

rcParams['figure.figsize'] = [8, 8]
plt.imshow(binary_skin, cmap = 'gray')
plt.title('Binary Depiction of 1st Image')
plt.axis('off')

plt.show()


##########################################################################################
# Opening & Closing Filters

opening_kernel = np.ones((3, 3), dtype = np.uint8)
opening = cv2.morphologyEx(binary_skin, cv2.MORPH_OPEN, opening_kernel)

closing_kernel = np.ones((30, 30), dtype = np.uint8)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, closing_kernel)

rcParams['figure.figsize'] = [10, 10]
plt.imshow(opening, cmap='gray')
plt.title('Opening Filter')
plt.axis('off')

plt.show()

plt.imshow(closing, cmap='gray')
plt.title('Closing Filter')
plt.axis('off')

plt.show()


##########################################################################################
# Find & distinguish the head and the hands

s = generate_binary_structure(2, 2)
labeled_array, num_features = label(closing, structure = s)
print()

print('Number of Distinguished Parts = ', num_features, '(Head, Right Hand, Left Hand)')
print()

rcParams['figure.figsize'] = [5, 5]
plt.imshow(labeled_array)
plt.axis('off')
plt.title('Distinguished Parts')

plt.show()


##########################################################################################
# Part 1.2: Monitoring of Face and Hands 
##########################################################################################

##########################################################################################
# Function for bounding box (delimitation)

def bounding_box(image, val):

    threshold_bin = (image==val).astype('uint8')
    
    contours = cv2.findContours(threshold_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 2:
        contours = contours[0] 
    else:
        contours[1]

    for counter in contours:
        x, y, width, height = cv2.boundingRect(counter)
    
    return x, y, width, height


##########################################################################################
# Calculation of each body part

x, y, width, height = bounding_box(labeled_array, 1)
print('Head Part: x =', x, ', y =', y, ', Width =', width, ', Height =', height)

x, y, width, height = bounding_box(labeled_array, 2)
print('Left Hand Part: x =', x, ', y =', y, ', Width =', width, ', Height =', height)

x, y, width, height = bounding_box(labeled_array, 3)
print('Right Hand Part: x =', x, ', y =', y, ', Width =', width, ', Height =', height)

print()


##########################################################################################
# Part 1.2.1: Implementation of Lucas-Kanade Algorithm 
##########################################################################################

##########################################################################################
# Lucas-Kanade Algorithm

def lucas_kanade(I1, I2, features, rho, epsilon, dx0, dy0, converge = 0.02):
    
    max_iterations = 50
    dx, dy = [], []
    
    I1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
    I1 = I1/np.max(I1)

    I2 = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)
    I2 = I2/np.max(I2)
    
    x0, y0 = np.meshgrid(np.arange(0, I1.shape[1]), np.arange(0, I1.shape[0]))
    
    gradient = np.gradient(I1)
    A1, A2 = gradient[1], gradient[0]
    
    kernelSize = int (np.ceil(3*rho)*2 + 1)
    G = cv2.getGaussianKernel(kernelSize, rho)
    G = G @ G.T
    
    for pixel in features:
        dx0, dy0 = 0, 0
        x, y = int(pixel[0][1]), int(pixel[0][0])

        for i in range(max_iterations):

            I1s = map_coordinates(I1,[np.ravel(y0+dy0), np.ravel(x0+dx0)], order=1).reshape(I1.shape)
            A1s = map_coordinates(A1,[np.ravel(y0+dy0), np.ravel(x0+dx0)], order=1).reshape(A1.shape)
            A2s = map_coordinates(A2,[np.ravel(y0+dy0), np.ravel(x0+dx0)], order=1).reshape(A2.shape)
            E = I2 - I1s

            a11 = cv2.filter2D(A1s**2,  -1, G)[x,y] + epsilon
            a12 = cv2.filter2D(A1s*A2s, -1, G)[x,y]
            a21 = cv2.filter2D(A1s*A2s, -1, G)[x,y]
            a22 = cv2.filter2D(A2s**2,  -1, G)[x,y] + epsilon
            determinant = a11*a22-a12*a21
            
            b1 = cv2.filter2D(A1s*E, -1, G)[x,y]
            b2 = cv2.filter2D(A2s*E, -1, G)[x,y]
            
            u1 = (a22*b1 - a12*b2)/determinant
            u2 = (-a21*b1 + a11*b2)/determinant
            dx0 = dx0 + u1
            dy0 = dy0 + u2

            if (np.linalg.norm(u1)<converge and np.linalg.norm(u2)<converge):
                break
        
        dx.append(dx0)
        
        dy.append(dy0)
    
    return np.array(dx), np.array(dy)


##########################################################################################
# Shi-Tomasi Corner Detection

def shi_tomasi(image, parameters, Harris):
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    corners = cv2.goodFeaturesToTrack(gray,**parameters, useHarrisDetector=Harris)
    
    return corners


##########################################################################################

# Find bounding boxes
head_dict = {'x':138, 'y':88, 'width':73, 'height':123}
lefthand_dict = {'x':47, 'y':243, 'width':71, 'height':66}
righthand_dict = {'x':162, 'y':264, 'width':83, 'height':48}

# Read 1st Image
I1 = np.array(cv2.imread('GreekSignLanguage/1.png', 1))
# Crop Image
head_box1 = I1[head_dict['y']:head_dict['y']+head_dict['height'], head_dict['x']:head_dict['x']+head_dict['width']]
lefthand_box1 = I1[lefthand_dict['y']:lefthand_dict['y']+lefthand_dict['height'], lefthand_dict['x']:lefthand_dict['x']+lefthand_dict['width']]
righthand_box1 = I1[righthand_dict['y']:righthand_dict['y']+righthand_dict['height'], righthand_dict['x']:righthand_dict['x']+righthand_dict['width']]

# Read 2nd Image
I2 = np.array(cv2.imread('GreekSignLanguage/2.png', 1))
# Crop Image
head_box2 = I2[head_dict['y']:head_dict['y']+head_dict['height'], head_dict['x']:head_dict['x']+head_dict['width']]
lefthand_box2 = I2[lefthand_dict['y']:lefthand_dict['y']+lefthand_dict['height'], lefthand_dict['x']:lefthand_dict['x']+lefthand_dict['width']]
righthand_box2 = I2[righthand_dict['y']:righthand_dict['y']+righthand_dict['height'], righthand_dict['x']:righthand_dict['x']+righthand_dict['width']]


##########################################################################################
# Implementation of Boxes & Plotting

def plot_boxes(image, head_dict, lefthand_dict, righthand_dict, save=False, directory='./', name='image'):
    
    head_rect = plt.Rectangle((head_dict['x'], head_dict['y']), head_dict['width'], head_dict['height'], edgecolor='r', facecolor='none')
    left_rect = plt.Rectangle((lefthand_dict['x'], lefthand_dict['y']), lefthand_dict['width'], lefthand_dict['height'], edgecolor='c', facecolor='none')
    right_rect = plt.Rectangle((righthand_dict['x'], righthand_dict['y']), righthand_dict['width'], righthand_dict['height'], edgecolor='m', facecolor='none')

    fig, ax = plt.subplots()
    
    rcParams['figure.figsize'] = [6, 6]
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.add_patch(head_rect)
    ax.add_patch(left_rect)
    ax.add_patch(right_rect)
    plt.axis('off')
    plt.title('Bounding Boxes of Image '+name)
    
    if save==True:
        title = directory + name + '.png'
        plt.savefig(title)

    plt.show()


##########################################################################################
# Implementation of Optical Flow & Plotting

def plot_optical_flow(features, dx, dy, frame, box, name, save=False, directory='./'):
    
    features = features.astype('int').reshape(features.shape[0], features.shape[2])
    
    rcParams['figure.figsize'] = [6,6]
    plt.quiver(features[:,0], features[:,1],-dx, -dy, angles='xy', scale=100)
    
    title = 'Image '+str(frame)+': Optical Flow of '+box+'.'
    plt.title(title)
    
    if save==True:
        title = directory +'/'+ name + '.png'
        plt.savefig(title)
    
    plt.show()


# Run & Save in motion_tracking folder
plot_boxes(I1, head_dict, lefthand_dict, righthand_dict, False ,directory='./motion_tracking/', name='1')


##########################################################################################

rho = 8
epsilon = 0.01
dx0 = 0
dy0 = 0
max_corners = 100
mindistance = 15
quality = 0.01

parameters  = dict(maxCorners = max_corners, qualityLevel = quality, minDistance = mindistance)
parameters2 = dict(maxCorners = 200, qualityLevel = 0.01, minDistance = 5)

# Lucas Kanade on Head Bounding Box
corners_head = shi_tomasi(head_box2, parameters, False)
dx1, dy1 = lucas_kanade(head_box1, head_box2, corners_head, rho, epsilon, 0,0)
plot_optical_flow(corners_head, dx1, dy1, frame='1', box='Head', name='1', save=False, directory='optical_flow/head')

# Lucas Kanade on Left Hand Bounding Box
corners_left = shi_tomasi(lefthand_box2, parameters2, False)
dx2, dy2 = lucas_kanade(lefthand_box1, lefthand_box2, corners_left, rho, epsilon, 0,0)
plot_optical_flow(corners_left, dx2, dy2, frame='1', box='Left Hand', name='1', save=False, directory='optical_flow/left_hand')

# Lucas Kanade on Left Hand Bounding Box
corners_right = shi_tomasi(righthand_box2, parameters2, False)
dx3, dy3 = lucas_kanade(righthand_box1, righthand_box2, corners_right, rho, epsilon, 0,0)
plot_optical_flow(corners_right, dx3, dy3, frame='1', box='Right Hand', name='1', save=False, directory='optical_flow/right_hand')


##########################################################################################
# Part 1.2.2: Calculation of Window Displacement by Optical Flow Vectors 
##########################################################################################

##########################################################################################

# Function that returns mean speed vectors dx, dy of the bounding box

def displ(dx, dy, threshold=0.5):
    
    energy = dx**2 + dy**2
    
    mask = (energy > threshold).astype('int32')
    
    if np.sum(mask)==0:
        return 0,0
    
    dx_mean = np.sum(dx*mask)/np.sum(mask)
    
    dy_mean = np.sum(dy*mask)/np.sum(mask)

    return int(np.round(dx_mean)), int(np.round(dy_mean))

# Experimentation

epsilons = [0.01, 0.05]
rhos = [1, 3, 5]
experiments = []
thresholds = [0.001, 0.2, 0.5, 0.7]

for e in epsilons:
    for r in rhos:
        dx, dy = lucas_kanade(lefthand_box1, lefthand_box2, corners_left, r, e, 0,0)
        
        experiments.append((dx, dy))

# Print Total Displacements

counter = 0
for k in range(4):
    for i in range(2):
        for j in range(3):
            
            dispx, dispy = displ(-experiments[counter][0], -experiments[counter][1], threshold=thresholds[k])
            
            print('epsilon = '+str(epsilons[i])+", rho = "+str(rhos[j])+", Threshold = "+str(thresholds[k])+": dx = "+str(dispx)+", dy = "+str(dispy) )
            
            counter=+1
    
    print()


##########################################################################################
# Application for Lucas-Kanade Function can be seen in the Appendix of the Report
##########################################################################################


##########################################################################################
# Part 1.2.3: Multi-Scale Optical Flow Calculation 
##########################################################################################

##########################################################################################
# Implementation of Multiscale Lucas-Kanade Algorithm
def multi_lk(I1, I2, features, rho, epsilon, scale, parameters):
    
    if scale == 0:
        
        dx, dy = lucas_kanade(I1, I2,features, rho, epsilon, 0, 0)
        return dx, dy
        
    else:
        
        Gr = cv2.getGaussianKernel(3, rho)
        I1 = cv2.filter2D(I1, -1, Gr)
        I2 = cv2.filter2D(I2, -1, Gr)
        
        faceOld = cv2.resize(I1, (I1.shape[1]//2, I1.shape[0]//2), interpolation=cv2.INTER_CUBIC)
        faceNew = cv2.resize(I2, (I2.shape[1]//2, I2.shape[0]//2), interpolation=cv2.INTER_CUBIC)
        
        new_features = features//2 -1
        
        dx0, dy0 = multi_lk(faceOld, faceNew, new_features, rho, epsilon, scale-1, parameters)
        dx, dy = lucas_kanade(I1, I2, features, rho, epsilon, 2*dx0, 2*dy0)

        return dx, dy

##########################################################################################
# Application for Multiscale Lucas-Kanade Function can be seen in the Appendix of the Report
##########################################################################################



##########################################################################################
#######################################   END OF PART 1   ################################
##########################################################################################