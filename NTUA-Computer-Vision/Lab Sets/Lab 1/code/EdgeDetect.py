
import numpy as np
import utils
from scipy import ndimage
from scipy.signal import fftconvolve as conv2
from cv2 import erode as imerode
from cv2 import dilate as imdilate
from cv2 import BORDER_REFLECT
from cv2 import filter2D
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def EdgeDetect(I, sigma, theta_edge, laplac_type='linear'):
    """
        Step 1.2 - Function to detect edges of an Image
        Args:
            I: np.array of image
            sigma: Variance of noise
            theta_edge: Threshold Parameter
            laplacian_approx: Parameter to determine LoG approximation
            choices:
                linear: Perform LoG convolution
                morph: Use morphological filters
    """

    ''' Step 1.2.1 - Create Filters '''

    kernel_size = np.ceil(3 * sigma) * 2 + 1
    filter_size = (kernel_size, kernel_size)
    LoG, (x, y) = utils.fspecial(name='laplacian_of_gaussian',
                                 shape=filter_size, sigma=sigma) # XXX
    G_sigma, _ = utils.fspecial(name='gaussian', shape=filter_size,
                                sigma=sigma)

    ''' Create smoothened image '''
    # I_sigma = conv2(I, G_sigma, mode='same')
    I_sigma = filter2D(I, -1, G_sigma)
    if laplac_type == 'linear':
        ''' Step 1.2.2a - L1 Approach'''
        L = conv2(I, LoG, mode="same")

    elif laplac_type == 'morph':
        ''' Step 1.2.2b Morphological Filters Approach '''
        L = imdilate(I_sigma, utils.MorphologicalFilters.FIVE_POINT_FILTER) + \
            imerode(I_sigma, utils.MorphologicalFilters.FIVE_POINT_FILTER) - \
            2 * I_sigma

    ''' Detect crossings and apply thresholding '''
    result = utils.filter_crossings(I_sigma, L,
                                    utils.MorphologicalFilters.FIVE_POINT_FILTER,
                                    theta_edge)

    return result
