# COMPUTER VISION
# LAB PROJECT #2
# SUBJECT: Optical Flow Estimation & Feature Extraction in Videos for Movement Recognition


##########################################################################################
# Libraries & necessary packets
##########################################################################################


import os
from os import listdir
from os.path import isfile, join
from cv21_lab2_2_utils import read_video, show_detection, orientation_histogram, bag_of_words, svm_train_test
import numpy as np
import cv2
import scipy
from scipy import signal
from scipy import ndimage as nd

import warnings
warnings.filterwarnings('ignore')


##########################################################################################
# Part 2: Spacio-Temporal Interest Points Detection and Feature Extraction in Human Action Videos
##########################################################################################

##########################################################################################
# Part 2.1: Spatio-Temporal Points of Interest 
##########################################################################################

##########################################################################################
# input videos

boxing_videos  = ['./data/boxing/'  +f for f in os.listdir('./data/boxing/')  if isfile(join('./data/boxing/' , f))]
running_videos = ['./data/running/' +f for f in os.listdir('./data/running/') if isfile(join('./data/running/', f))]
walking_videos = ['./data/walking/' +f for f in os.listdir('./data/walking/') if isfile(join('./data/walking/', f))]

boxing_video1  = read_video(boxing_videos[1],  200)
boxing_video2  = read_video(boxing_videos[4],  200)
boxing_video3  = read_video(boxing_videos[8],  200)
running_video1 = read_video(running_videos[1], 200)
running_video2 = read_video(running_videos[4], 200)
running_video3 = read_video(running_videos[8], 200)
walking_video1 = read_video(walking_videos[1], 200)
walking_video2 = read_video(walking_videos[4], 200)
walking_video3 = read_video(walking_videos[8], 200)


##########################################################################################
# Part 2.1.1 - Harris Detector 
##########################################################################################

##########################################################################################
# Implementation of 1D Gaussian

def Gaussian_1D (sigma):
    
    kernelSize = int (np.ceil(3*sigma)*2 + 1)
    
    G = cv2.getGaussianKernel(kernelSize, sigma)
    
    return G

# Implementation of 2D Gaussian

def Gaussian_2D (sigma):
    
    G_1D = Gaussian_1D(sigma)
    
    G_2D = G_1D @ G_1D.transpose()
    
    return G_2D

##########################################################################################
# Implementation of Harris-Stephens detector

def HarrisStephensDetector (V, sigma, tau, k, s):
    
    V = V.astype(np.float32)
    
    G_space = Gaussian_2D(sigma)
    G_space = G_space.reshape(G_space.shape[0], G_space.shape[1], 1)
    G_time = Gaussian_1D(tau)
    G_time = G_time.reshape(1, 1, G_time.shape[0])
    G_spatio_temporal = nd.convolve(G_time, G_space)
    
    dx_filter = np.array([[[-1], [0], [1]]], dtype=np.int8)
    dy_filter = np.array([[[-1]], [[0]], [[1]]], dtype=np.int8)
    dt_filter = np.array([[[-1,  0,  1]]], dtype=np.int8)
    
    L = nd.convolve(V, G_spatio_temporal)
    Lx = nd.convolve(L, dx_filter)
    Ly = nd.convolve(L, dy_filter)
    Lt = nd.convolve(L, dt_filter)
    Lxx = Lx*Lx
    Lyy = Ly*Ly
    Ltt = Lt*Lt
    Lxy = Lx*Ly
    Lxt = Lx*Lt
    Lyt = Ly*Lt
    
    # smoothing
    g_space = Gaussian_2D(s*sigma)
    g_space = g_space.reshape(g_space.shape[0], g_space.shape[1], 1)
    g_time = Gaussian_1D(s*tau)
    g_time = g_time.reshape(1, 1, g_time.shape[0])
    g_spatio_temporal = nd.convolve(g_time, g_space)
    
    Mxx = nd.convolve(Lxx, g_spatio_temporal)
    Myy = nd.convolve(Lyy, g_spatio_temporal)
    Mtt = nd.convolve(Ltt, g_spatio_temporal)
    Mxy = nd.convolve(Lxy, g_spatio_temporal)
    Mxt = nd.convolve(Lxt, g_spatio_temporal)
    Myt = nd.convolve(Lyt, g_spatio_temporal)
    
    # Harris-Stephens
    trace = Mxx + Myy + Mtt
    det = (Mxx*Myy*Mtt - Mxx*Myt**2) - (Mtt*Mxy**2 - Mxy*Mxt*Myt) + (Mxy*Mxt*Myt - Myy*Mxt**2)
    H = det - k*trace**3
    
    return H


##########################################################################################
# Part 2.1.2 - Gabor Detector 
##########################################################################################

##########################################################################################
# Implementation of Gabor detector

def GaborDetector (V, sigma, tau):
    
    V = V.astype(np.float32)
    
    # smoothing
    G_space = Gaussian_2D(sigma)
    G_space = G_space.reshape(G_space.shape[0], G_space.shape[1], 1)
    
    Vs = nd.convolve(V, G_space)
    
    # Gabor
    w = 4/tau
    
    time = np.linspace(int(-2*tau), int(2*tau), int(4*tau+1), endpoint=True)
    
    h_ev = -np.cos(2*np.pi*time*w)*np.exp((-time**2)/(2*tau**2))
    h_ev = h_ev/np.linalg.norm(h_ev, ord=1)
    h_ev = h_ev.reshape(1, 1, h_ev.shape[0])

    h_od = -np.sin(2*np.pi*time*w)*np.exp((-time**2)/(2*tau**2))
    h_od = h_od/np.linalg.norm(h_od, ord=1)
    h_od = h_od.reshape(1, 1, h_od.shape[0])
    
    H = (nd.convolve(Vs, h_ev))**2 + (nd.convolve(Vs, h_od))**2
    
    return H


##########################################################################################
# Part 2.1.3 - Importance Crieterion & Experimentation 
##########################################################################################

##########################################################################################
# This function returns matrix with the interest points

def calculate_interest_points(H, N, sigma):
    
    N_largest = np.dstack(np.unravel_index(np.argsort(H.ravel()), (H.shape[0], H.shape[1], H.shape[2])))
    
    N_largest = N_largest.reshape(N_largest.shape[1], N_largest.shape[2])
    
    N_largest = N_largest[-N:]
    
    Output = np.array([np.append(pair[:2][::-1],[pair[2], sigma]) for pair in N_largest])
        
    return Output

##########################################################################################
# Implementation of some experiments can be seen in the Appendix of the Report
##########################################################################################


##########################################################################################
# Part 2.2: Spatio-Temporal Histographic Descriptors 
##########################################################################################

##########################################################################################
# Part 2.2.1 - Calculation of Gradient and Optical Flow
##########################################################################################

##########################################################################################
# Implementation of Gradient

def gradient_implem(V):
    
    dy, dx, _ = np.gradient(V)
    
    return (dy, dx)

# Implementation of Optical Flow

def optical_flow(V):
    
    V = V.astype(np.uint8)
    
    flow_x = np.zeros((V.shape[0], V.shape[1], V.shape[2]))
    
    flow_y = np.zeros((V.shape[0], V.shape[1], V.shape[2]))

    for i in range(V.shape[2]):
        tmp = i
        
        if i == V.shape[2]-1:
            tmp = i-1
                
        temp = cv2.optflow.DualTVL1OpticalFlow_create(nscales=1).calc(V[:,:,tmp], V[:,:,tmp+1], None)
        
        flow_x[:,:,tmp] = temp[:,:,1]
        flow_y[:,:,tmp] = temp[:,:,0]
        
    print(u'\u2713', end='')

    return (flow_y, flow_x)


##########################################################################################
# Part 2.2.2 - HOG & HOF descriptors
##########################################################################################

##########################################################################################
# Implementation of HOG/HOF descriptors

def HOG_HOF_Descriptor (V, points, descriptor, nbins=9, n=3, m=3):
    
    descr = []
    height = V.shape[0]
    width  = V.shape[1]
    
    if descriptor == 'HOF':
        dy, dx = optical_flow(V)

    elif descriptor == 'HOG':
        dy, dx = gradient_implem(V)

    elif descriptor == 'HOG/HOF':
        HOF = HOG_HOF_Descriptor(V, points, 'HOF', nbins, n, m)
        HOG = HOG_HOF_Descriptor(V, points, 'HOG', nbins, n, m)
        
        return np.concatenate((HOG, HOF))

    else:
        print('Error: Invalid Descriptor Type')
        
        return
    
    for point in points:
        
        x = point[0]
        y = point[1]
        t = point[2]
        
        sigma = point[3]
        side = int (np.round(4*sigma))
        
        x_right = min(width, x+side+1)
        x_left  = max(0, x-side)
        
        y_bot   = min(height, y+side+1)
        y_top   = max(0, y-side)
        
        Gx = dx[y_top:y_bot, x_left:x_right, t]
        Gy = dy[y_top:y_bot, x_left:x_right, t]
        
        descr.append(orientation_histogram(Gx, Gy, nbins, np.array([n,m])))
    
    return np.array(descr)


##########################################################################################
# Part 2.3: Bag of Visual Words Construction and use of Support Vector Machines for action classification 
##########################################################################################

##########################################################################################
# Part 2.3.1 - Dividing data in train set and test set
##########################################################################################

##########################################################################################
# Implementation of Division of Data

def division_of_data(training_videos_file):

    temp_file = open('./data/' + training_videos_file, 'r')
    
    lines = temp_file.readlines()
    train_names = [i.strip() for i in lines]

    train_set_temp = []
    train_tags = []
    test_set_temp = []
    test_tags = []

    for name in os.listdir('./data/boxing/'):
        if name in train_names:
            train_set_temp.append(read_video('./data/boxing/'+name, 200))
            train_tags.append(1)

        # elif name != '.DS_Store':
        else:
            test_set_temp.append(read_video('./data/boxing/'+name, 200))
            test_tags.append(1)
    
    for name in os.listdir('./data/running/'):
        if name in train_names:
            train_set_temp.append(read_video('./data/running/'+name, 200))
            train_tags.append(0)

        else:
            test_set_temp.append(read_video('./data/running/'+name, 200))
            test_tags.append(0)
    
    for name in os.listdir('./data/walking/'):
        if name in train_names:
            train_set_temp.append(read_video('./data/walking/'+name, 200))
            train_tags.append(2)

        else:
            test_set_temp.append(read_video('./data/walking/'+name, 200))
            test_tags.append(2)

    train_set = np.array(train_set_temp)

    test_set = np.array(test_set_temp)
    
    return train_set, test_set, train_tags, test_tags


##########################################################################################
# Part 2.3.2 - Bag of Visual Words Global Representation
##########################################################################################

##########################################################################################
# Implementation of Descriptor for Video Set

def descriptor_for_video_set(descriptor, detector, video_set, sigma, tau, k, s):
    
    nbins = 9
    descriptors_for_video_set = []
    n = int(np.round(4*sigma))
    m = int(np.round(4*sigma))

    if descriptor != 'HOF' and descriptor != 'HOG' and descriptor != 'HOG/HOF':
        print('Error: Invalid Descriptor')
        return
    
    if detector == 'Harris':
        for video in video_set:
            
            criterion = HarrisStephensDetector(video, sigma, tau, k, s)
            
            points = calculate_interest_points(criterion, 600, sigma)
            
            descr = HOG_HOF_Descriptor(video, points, descriptor, nbins, n, m)
            
            descriptors_for_video_set.append(descr)
            
    elif detector == 'Gabor':
        for video in video_set:
            
            criterion = GaborDetector(video, sigma, tau)
            
            points = calculate_interest_points(criterion, 600, sigma)
            
            descr = HOG_HOF_Descriptor(video, points, descriptor, nbins, n, m)
            
            descriptors_for_video_set.append(descr)
            
    else:
        print('Error: Invalid Detector')
        
        return
    
    return descriptors_for_video_set

# Implementation of BoVW Function

def BoVW_implem(train_set, test_set, descriptor, detector, sigma, tau, k, s, num_centers):
    
    descr_train = descriptor_for_video_set(descriptor, detector, train_set, sigma, tau, k, s)
    
    descr_test  = descriptor_for_video_set(descriptor, detector, test_set,  sigma, tau, k, s)
    
    bovw_train, bovw_test = bag_of_words(descr_train, descr_test, num_centers)
    
    return (bovw_train, bovw_test)


##########################################################################################
# Part 2.3.3 - Classification using Support Vector Machine (SVM)
##########################################################################################

##########################################################################################
# Implementation of SVM classification

def SVM_classification(train_set, test_set, train_tags, test_tags, descriptor, detector, sigma, tau, k, s, num_centers=20):
    
    bovw_train, bovw_test = BoVW_implem(train_set, test_set, descriptor, detector, sigma, tau, k, s, num_centers)
    
    accuracy, pred = svm_train_test(bovw_train, train_tags, bovw_test, test_tags)

    print("Classification with the following parameters:")
    print("\nDetector: {},".format(detector), end=' ')
    print("\nDescriptor: {},".format(descriptor), end=' ')
    print("\nsigma = {0:.2f}, \ntau = {1:.2f}".format(sigma, tau))
    print()

    print("\nClassification terminated. Results:")
    print("\nAccuracy = {0:.2f}%".format(accuracy*100))
    print("\nPred = ", end="")
    print(pred)
    print()

    return


##########################################################################################
# Part 2.3.4 - Experiments with different detector-descriptor combinations
##########################################################################################

##########################################################################################
# typical values for tests

typ = { 'sigma': 4, 'tau': 1.5, 'k': 0.005, 's': 2 }

# Implementation of experiments

train_set, test_set, train_tags, test_tags = division_of_data('training_videos.txt')

# typical values
num_centers = 20
sigma = typ['sigma']
tau = typ['tau']
k = typ['k']
s = typ['s']

# Detector: Harris,  Descriptor: HOG
print('Classification 1')
SVM_classification(train_set, test_set, train_tags, test_tags, 'HOG', 'Harris', sigma, tau, k, s, 20)
print()

# Detector: Harris,  Descriptor: HOF
print('Classification 3')
SVM_classification(train_set, test_set, train_tags, test_tags, 'HOF', 'Harris', sigma, tau, k, s, 20)
print()

# Detector: Harris,  Descriptor: HOG/HOF
print('Classification 5')
SVM_classification(train_set, test_set, train_tags, test_tags, 'HOG/HOF', 'Harris', sigma, tau, k, s, 20)
print()

# Detector: Gabor,   Descriptor: HOG
print('Classification 2')
SVM_classification(train_set, test_set, train_tags, test_tags, 'HOG', 'Gabor', sigma, tau, k, s, 20)
print()

# Detector: Gabor,   Descriptor: HOF
print('Classification 4')
SVM_classification(train_set, test_set, train_tags, test_tags, 'HOF', 'Gabor', sigma, tau, k, s, 20)
print()

# Detector: Gabor,   Descriptor: HOG/HOF
print('Classification 6')
SVM_classification(train_set, test_set, train_tags, test_tags, 'HOG/HOF', 'Gabor', sigma, tau, k, s, 20)
print()


##########################################################################################
# Part 2.3.5 - Experiments with different data partitions can be seen in the Appendix of the Report
##########################################################################################


##########################################################################################
#######################################   END OF PART 2   ################################
##########################################################################################