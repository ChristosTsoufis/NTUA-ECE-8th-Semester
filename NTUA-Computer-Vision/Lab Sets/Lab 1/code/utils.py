
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from cv2 import imread, COLOR_BGR2RGB, COLOR_RGB2GRAY, cvtColor, Sobel, CV_32F
from cv2 import erode as imerode
from cv2 import dilate as imdilate
from cv2 import KeyPoint
from cv2 import getRotationMatrix2D
from cv2 import warpAffine
from scipy.signal import fftconvolve as conv2
from cv2 import cartToPolar
import pickle


class MorphologicalFilters:
    FIVE_POINT_FILTER = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                                 dtype=np.uint8)


def get_gaussian_filter(sigma):
    '''Return an isotropic gaussian filter G_σ
        Args:
            sigma: The Gaussian variance
    '''
    kern_size = np.ceil(3 * sigma) * 2 + 1
    filter_size = (kern_size, kern_size)
    G, _ = fspecial('gaussian', filter_size, sigma=sigma)
    return G, kern_size


def imnoise(I, PSNR):
    """ Add AGWN to images
        Args:
            X: The image as np.array
            PSNR: Peak SNR to determine variance
        Returns:
            np.array: Image plus AGWN
    """

    I_max = np.max(I)
    I_min = np.min(I)

    sigma = (I_max - I_min) / 10**(PSNR / 20)

    n = np.random.normal(0, sigma, size=I.shape)

    return I + n, sigma


def get_grid(shape):
    '''Returns a meshgrid of m x n shape
        Args:
            shape: The shape of the grid
    '''
    m, n = [(ss-1.)/2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    return (y, x)


def fspecial(name, shape, normalize=True, **kwargs):
    ''' Generates special functions resembling MATLAB's fspecial'''
    if name == 'gaussian':
        sigma = kwargs['sigma']
        y, x = get_grid(shape)

        h = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
        h[h < np.finfo(h.dtype).eps*h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh

        return h, (x, y)
    elif name == 'laplacian_of_gaussian':
        sigma = kwargs['sigma']
        y, x = get_grid(shape)
        std2 = sigma**2
        h = np.exp(-(x*x + y*y)/(2*sigma**2))
        h[h < np.finfo(h.dtype).eps*h.max()] = 0
        sumh = h.sum()

        if sumh != 0:
            h /= sumh

        h1 = h * (x*x + y*y - 2*std2) / (std2**2)
        h1sum = h1.sum()
        h = h1 - h1sum / (shape[1] * shape[0])
        return h, (x, y)
    else:
        raise NotImplementedError("Not supported yet.")


def surf(X, Y, Z, **kwargs):
    '''
        Surface plot of Z = f(X, Y)
        Args:
            X, Y: Spatial Coordinates
            Z: Values of Z = f(X, Y)
    '''
    fig = plt.figure()
    if 'title' in kwargs:
        fig.title(kwargs['title'])
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)


def binary_sign(L):
    '''
        Step 1.2.3a - Returns the binary sign image of an input image L
        Args:
            L: np.array image
    '''
    return (L >= 0).astype(np.float64)


def boundary_operator(X, B):
    '''
        Step 1.2.3.b
        Return the morphological boundary operator which is defined as
        Args:
            X: The binary sign image
            B: Morphological Filter (Kernel)
    '''

    return imdilate(X, B) - imerode(X, B)


def zero_crossings(L, B):
    '''
        Step 1.2.3 - Return the zero crossings of an image
        Args:
            L: The filtered image
            B: The morphological filter
    '''
    X = binary_sign(L)
    Y = boundary_operator(X, B)
    return Y == 1


def filter_crossings(I, L, B, theta_edge):
    '''
        Step 1.2.4 - Filter crossings by thresholding
        Args:
            I: Original (Smoothened) Image
            L: Filtered Image
            B: Morphological Filter
            threshold: Threshold value
    '''
    Y = zero_crossings(L, B)
    dx, dy = np.gradient(I)
    grad_norm = np.sqrt(dx**2 + dy**2)
    max_grad = np.max(grad_norm)
    Q = grad_norm > theta_edge * max_grad

    return (Q & Y).astype(np.float64)


def quality_factor(T, D):
    '''
        Step: 1.3.2 - Calculate precision, recall for edge detection
        Args:
            T: golden binary image array
            D: binary image array
    '''
    A = T == 1.0
    B = D == 1.0
    cnt = (A & B).sum()
    P = cnt / A.sum()
    R = cnt / B.sum()
    return 0.5 * (P + R)


def structural_eigenvalues(J1, J2, J3):
    '''
        Structural eigenvalues λ+,λ- for the Harris-Stephens
        Corner detector
        Args:
            J1, J2, J3: Structural Tensors
    '''
    x1 = 1/2 * (J1 + J3 + np.sqrt((J1 - J3)**2 + 4 * J2**2))
    x2 = 1/2 * (J1 + J3 - np.sqrt((J1 - J3)**2 + 4 * J2**2))
    return x1, x2


def structural_tensor(I_sigma, G):
    ''' Step: 2.1.1: Structural tensor '''
    def get_d_multiples(dx, dy):
        dx_dx = dx * dx
        dx_dy = dx * dy
        dy_dy = dy * dy
        return dx_dx, dx_dy, dy_dy
    dims = len(I_sigma.shape)
    if dims == 3:
        # I_sigma shape: (nB, nB, 3)
        levels = I_sigma.shape[2]
        J1 = np.zeros(shape=I_sigma.shape, dtype=np.float64)
        J2 = np.zeros(shape=I_sigma.shape, dtype=np.float64)
        J3 = np.zeros(shape=I_sigma.shape, dtype=np.float64)
        for level in range(levels):
            dx, dy = np.gradient(I_sigma[:, :, level])
            dx_dx, dx_dy, dy_dy = get_d_multiples(dx, dy)
            J1[:, :, level] = conv2(dx_dx, G, mode='same')
            J2[:, :, level] = conv2(dx_dy, G, mode='same')
            J3[:, :, level] = conv2(dy_dy, G, mode='same')
        return J1, J2, J3
    elif dims == 2:
        # I_sigma shape: (nB, nB)
        dx, dy = np.gradient(I_sigma)
        dx_dx, dx_dy, dy_dy = get_d_multiples(dx, dy)
        J1 = conv2(dx_dx, G, mode='same')
        J2 = conv2(dx_dy, G, mode='same')
        J3 = conv2(dy_dy, G, mode='same')
        return J1, J2, J3
    else:
        raise ValueError('Wrong dimensions')


def disk(n):
    '''
        Return Disk structural element
        Args:
            n: Kernel size
    '''
    r = n // 2
    y, x = get_grid(shape=(n, n))
    nhood = x**2 + y**2 <= r**2
    return nhood.astype(np.uint8)


def local_maxima(ns, R):
    '''
        Returns local_maxima in a square window.
        Args:
            ns: window_size/2
            R: array
        Returns: bool array, with True in local_maxima
    '''
    B_sq = disk(ns)
    return R == imdilate(R, B_sq)


def filter_maxima(R, threshold):
    '''
        Filters out elements with low values
        Args:
            R: array
            threshold: a threshold value
        Returns: bool array, with True in positions where the element is
        > threshold * R.max
    '''
    R_max = np.max(R)
    return R > threshold * R_max


def cornerness_criterion(R, ns, theta_corn, plot=False, sigma=None):
    ''' Cornerness Criterion for Harris-Stephens Corner Detector
        and blob detector (Hessian)
        Args:
            R: R matrix (det of Hessian of det(M) - k^2 tr^2(M))
            ns: Structural element size
            theta_corn: Threshold value
    '''
    cond_1 = local_maxima(ns, R)
    cond_2 = filter_maxima(R, theta_corn)
    if plot and (sigma is not None):
        plt.figure()
        plt.imshow(brightness(R), cmap=plt.get_cmap('gray'))
        plt.title('Cornerness Criterion for σ = {}'.format(sigma))

    return cond_1 & cond_2


def conv3(I, h):
    '''
        Multichannel convolution using FFT
        Args:
            I: Multichannel (RGB) image
            h: Filter (Linear Space-Invariant)
    '''
    levels = I.shape[2]
    result = np.zeros(shape=I.shape, dtype=np.float64)

    for level in range(levels):
        result[:, :, level] = conv2(I[:, :, level], h, mode='same')

    return result


def brightness(I):
    ''' Convert to grayscale '''
    if len(I.shape) == 2:
        return I
    return np.dot(I[:, :, :3], [0.2989, 0.5870, 0.1140])


def or_channels(I):
    ''' Perform bitwise OR over channels '''
    return np.max(I, axis=2)


def and_channels(I):
    ''' Perform bitwise AND over channels '''
    return np.min(I, axis=2)


def apply_channels(I, f, *args):
    """
        Apply function f accross all channels
    """
    levels = I.shape[2]
    result = np.zeros(shape=I.shape, dtype=np.float64)

    for level in range(levels):
        result[:, :, level] = f(I[:, :, level], *args)

    return result


def angle_criterion(eigs, k):
    ''' Returns det(M) - k tr^2(M) '''
    l_p, l_m = eigs
    return l_p * l_m - k * (l_p + l_m)**2


def hessian_matrix(I_sigma):
    ''' Return Hessian matrix of an image '''
    def get_ch_gradients(I):
        dx, dy = np.gradient(I)
        dxx, dxy = np.gradient(dx)
        dyx, dyy = np.gradient(dy)
        return dxx, dxy, dyx, dyy
    if len(I_sigma.shape) == 3:
        I_sigma = I_sigma.transpose(2, 0, 1)
        _, _1, _2 = I_sigma.shape
        hessian = np.empty([3, 2, 2, _1, _2])
        for ch, I in enumerate(I_sigma):
            dxx, dxy, dyx, dyy = get_ch_gradients(I)
            hessian[ch][:][:][:][:] = \
                np.array([[dxx, dxy], [dyx, dyy]])
        hessian = hessian.transpose(3, 4, 0, 1, 2)
    else:
        dxx, dxy, dyx, dyy = get_ch_gradients(I_sigma)
        hessian = np.array([[dxx, dxy], [dyx, dyy]]).transpose(2, 3, 0, 1)
    return hessian


def interest_points_visualization(I, binary_image, sigma, fig, ax,
                                  scale_fcn=lambda s: 3*s):
    ''' Visualize keypoints of an image
        Args:
            I: Image to plot results on
            binary_image: ROIs
            sigma: Scale
            fig, ax: Figure and axis
            scale_fcn: Radius of plotted circles
    '''
    ax.set_aspect('equal')
    ax.imshow(I)
    for x in range(binary_image.shape[0]):
        for y in range(binary_image.shape[1]):
            if binary_image[x, y]:
                circ = Circle((y, x), scale_fcn(sigma), edgecolor='g', fill=False)
                ax.add_patch(circ)
    return fig, ax


def multiscale_interest_points_visualization(I, corners, diff_scales, fig, ax,
                                             scale_fcn=lambda s: 3*s):
    ''' Interest points visualization for multiscale detector '''
    for i, sigma in enumerate(diff_scales):
        interest_points_visualization(I, corners[:, :, i], sigma, fig, ax,
                                      scale_fcn)
    return fig, ax


def export_interest_point(binary_image, scales, outfile):
    """
        Export ROIs to .csv to be plotted with interest_points_visualization.p
        Args:
            binary_image: np.array of bool
            scales: Can be scalar or a list of scales
            outfile: The output csv file    print(np.sum(binary_image))

    """

    if isinstance(scales, list):
        raise NotImplementedError("Not implemented")
    else:
        levels = binary_image.shape[2]
        with open(outfile, 'w+') as f:
            for level in range(levels):
                for x in range(binary_image.shape[0]):
                    for y in range(binary_image.shape[1]):
                        if binary_image[x, y, level]:
                            f.write('{} {} {}\n'.format(x, y, scales))


def multidim_cumsum(I):
    """
        Calculates multidimensional cumsum using np.cumsum
        Args:
            I: The input array
    """
    Ic = I
    for dim in range(len(I.shape)):
        Ic = np.cumsum(Ic, axis=dim)

    return Ic


def pad_image(I, pad_width, mode):
    tmp = []
    for level in range(I.shape[2]):
        tmp.append(np.pad(I[:, :, level], pad_width=pad_width, mode=mode))

    return np.array(tmp).transpose(1, 2, 0)


def fast_convolve_box_filters(I, kernel_size, box_filters, coeffs):
    '''
        Perform fast convolution using box filters
        Args:
            I: Image
            kernel_size: Kernel Size
            box_filters: Array of boxes [a, b] x [c, d]
            coeffs: Coefficients
    '''
    def _convolve_box_filter(B):
        a, b, c, d = B

        temp = np.zeros(shape=Ip.shape, dtype=np.float64)

        for x in range(kernel_size, Ip.shape[0] - kernel_size):
            for y in range(kernel_size, Ip.shape[1] - kernel_size):
                temp[x, y] = Ic[x - a, y - c] + Ic[x - b - 1, y - d - 1] \
                            - Ic[x - a, y - d - 1] - Ic[x - b - 1, y - c]

        return temp

    ''' Pad original image '''
    Ip = np.pad(I, pad_width=kernel_size, mode='symmetric')

    ''' Calculate Cumsums '''
    Ic = multidim_cumsum(Ip).astype(np.float64)

    '''Apply box filtering'''
    Iconvolved = np.zeros(shape=(Ip.shape), dtype=np.float64)

    '''Apply box filters sucessively '''
    for box_filter, coeff in zip(box_filters, coeffs):
        Ir = _convolve_box_filter(box_filter)
        Iconvolved += coeff * Ir

    return Iconvolved[kernel_size:-kernel_size, kernel_size:-kernel_size]


def get_gaussian_approximation_filters(filter_type, sigma):
    ''' Construction of box filters for a Gaussian Kernel
        Args:
            filter_type: Can be any of [xx, yy, xy, yx]
            sigma: Scale
    '''
    n = int(2 * np.ceil(3 * sigma) + 1)
    center = n // 2

    if filter_type == 'xx':
        h = int(4 * np.floor(n / 6) + 1)
        w = int(2 * np.floor(n / 6) + 1)
        ''' Outer rectangle '''
        a1 = 0
        b1 = n - 1
        c1 = center - h // 2
        d1 = center + h // 2

        ''' Inner Rectangle '''
        a2 = center - w // 2
        b2 = center + w // 2
        c2 = c1
        d2 = d1
        return n, [[a1, b1, c1, d1], [a2, b2, c2, d2]], [1, -3]
    elif filter_type == 'yy':
        w = int(4 * np.floor(n / 6) + 1)
        h = int(2 * np.floor(n / 6) + 1)
        ''' Outer rectangle '''
        a1 = center - w // 2
        b1 = center + w // 2
        c1 = 0
        d1 = n - 1

        ''' Inner Rectangle '''
        a2 = a1
        b2 = b1
        c2 = center - h // 2
        d2 = center + h // 2

        Z = np.zeros(shape=(n, n), dtype=int)
        for i in range(a1, b1 + 1):
            for j in range(c1, d1 + 1):
                Z[i, j] += 1
        for i in range(a2, b2 + 1):
            for j in range(c2, d2 + 1):
                Z[i, j] -= 3
        return n, [[a1, b1, c1, d1], [a2, b2, c2, d2]], [1, -3]
    elif filter_type == 'xy' or filter_type == 'yx':
        w = int(2 * np.floor(n / 6) + 1)

        ''' Upper Left '''
        a1 = center - 1 - w
        b1 = a1 + w
        c1 = a1
        d1 = b1

        ''' Lower Left '''
        a2 = center + 1
        b2 = a2 + w
        c2 = center - 1 - w
        d2 = c2 + w

        ''' Upper Right '''
        b3 = center - 1
        a3 = b3 - w
        c3 = center + 1
        d3 = c3 + w

        ''' Lower Right '''
        a4 = center + 1
        b4 = a4 + w
        c4 = a4
        d4 = b4

        return n, [[a1, b1, c1, d1], [a2, b2, c2, d2],
                   [a3, b3, c3, d3], [a4, b4, d4, c4]], [1, -1, -1, 1]


def hessian_approx(I, sigma):
    ''' Returns the Hessian Approximation for an Image
        The Hessian Matrix of an image is the Matrix
                H = [[Lxx, Lxy],
                    [Lyx, Lyy]]
        where Lxx, Lyy, Lxy, Lyx correspond to the second
        derivatives of a Gaussian Filtered Image
        Args:
            I: Input Image
            sigma: Scale
    '''
    def hessian_approx_2d(I, sigma):
        Dxx = get_gaussian_approximation_filters('xx', sigma)
        Dyy = get_gaussian_approximation_filters('yy', sigma)
        Dxy = get_gaussian_approximation_filters('xy', sigma)

        Lxx = fast_convolve_box_filters(I, *Dxx)
        Lyy = fast_convolve_box_filters(I, *Dyy)
        Lxy = fast_convolve_box_filters(I, *Dxy)
        return Lxx, Lyy, Lxy

    if len(I.shape) == 2:
        return hessian_approx_2d(I, sigma)
    else:
        Lxx = np.empty(I.shape)
        Lyy = np.empty(I.shape)
        Lxy = np.empty(I.shape)
        for i in range(I.shape[-1]):
            Lxx[:, :, i], Lyy[:, :, i], Lxy[:, :, i] = \
             hessian_approx_2d(I[:, :, i], sigma)
    return Lxx, Lyy, Lxy


def hessian_approx_criterion(I, sigma, theta_corn, alpha=0.9, plot=False):
    ''' Hessian Approximation for the R criterion using box filters
        for LoG filtering
        Args:
            I: Input Image
            sigma: Scale
    '''
    Lxx, Lyy, Lxy = hessian_approx(I, sigma)
    n = int(2 * np.ceil(3 * sigma) + 1)
    R = Lxx * Lyy - (alpha * Lxy)**2
    R_max = np.max(R)

    B_sq = disk(n)

    cond_1 = (R == imdilate(R, B_sq))
    cond_2 = (R > theta_corn * R_max)

    if plot:
        fig = plt.figure()
        ax = plt.subplot(221)
        plt.imshow(R, cmap=plt.get_cmap('gray'))
        plt.title('Cornerness Criterion with Hessian Approximation for σ = {} '
                  'and θ_corn = {}'.format(sigma, theta_corn))
        ax = plt.subplot(222)
        plt.imshow(Lxx, cmap=plt.get_cmap('gray'))
        plt.title('Lxx')
        ax = plt.subplot(223)
        plt.imshow(Lyy, cmap=plt.get_cmap('gray'))
        plt.title('Lyy')
        ax = plt.subplot(224)
        plt.imshow(Lxy, cmap=plt.get_cmap('gray'))
        plt.title('Lxy')

    return cond_1 & cond_2


def features_to_keypoints(binary_map, scales):
    ''' Converts binary map to cv2.KeyPoint
        Args:
            binary_map: A binary image with ROIs
            scales: An int or a list of scales
    '''
    def single_scale_features_to_keypoints(binary_map, scale):
        result = []
        for x in range(binary_map.shape[0]):
            for y in range(binary_map.shape[1]):
                if binary_map[x, y]:
                    kp = KeyPoint(x, y, scale)
                    result.append(kp)
        return result

    if isinstance(scales, list):
        result = []

        for i, scale in enumerate(scales):
            scale_p = single_scale_features_to_keypoints(binary_map[:, :, i],
                                                         scale)
            result.extend(scale_p)
        return result
    else:
        return single_scale_features_to_keypoints(binary_map, scales)


def get_image_affinely_transformed(I, scales, angles):
    ''' Transforms and scales image according to scales and angles
        Args:
            I: Input Image
            scales: An array of scales
            angles: An array of angles
    '''
    rows, cols = I.shape[0], I.shape[1]
    result = []
    for sigma in scales:
        for theta in angles:
            M = getRotationMatrix2D(((cols-1)/2.0, (rows-1)/2.0), theta, sigma)
            dst = warpAffine(I, M, (cols, rows))
            result.append(dst)

    return result


def dump(obj, filename):
    ''' Serialize object '''
    with open(filename, 'wb+') as f:
        pickle.dump(obj, f)


def load_pickle(filename):
    ''' Load object from pickle '''
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        return data


class HOG:
    '''HOG Feature descriptor. Calculate HOG Features
        (Histogram of Oriented Gradients) of an image.
        For taking the derivatives gx and gy of the image
        our implementation uses the Sobel Operator.
    '''

    def __init__(self, nbins=9, cell_size=4, block_size=8):
        '''HOG Descriptor Constructor
            Args:
                nbins: Number of bins
                cell_size: Cell size
                padding: Padding size
        '''
        self.nbins = nbins
        self.cell_size = cell_size
        self.block_size = block_size
        self.half_size = block_size // 2
        assert(360 % self.nbins == 0)
        self.step = 360 // self.nbins

    def compute(self, I, keypoints):
        '''Copmute HOG Features
            Args:
                I: Query Image
                keypoints: List of keypoints (ROIs)
        '''

        ''' Calculate image gradients '''
        I_g = cvtColor(I, COLOR_RGB2GRAY)
        I_min = np.min(I_g)
        I_max = np.max(I_g)
        I_norm = (I_g - I_min) / (I_max - I_min)
        gx = Sobel(I_g, CV_32F, 1, 0, ksize=1)
        gy = Sobel(I_g, CV_32F, 0, 1, ksize=1)
        mag, angle = cartToPolar(gx, gy, angleInDegrees=True)
        result = []

        for i, keypoint in enumerate(keypoints):
            ''' Calculate HOG for i-th keypoint '''
            x, y = keypoint.pt
            x = int(x)
            y = int(y)
            hog = []

            ''' Block of certain keypoint [a, b] x [c, d]'''
            a = x - self.half_size
            b = x + self.half_size
            c = y - self.half_size
            d = y + self.half_size

            ''' Ignore blocks out of valid regions '''
            if a < 0 or c < 0 or b >= I.shape[0] or d >= I.shape[1]:
                continue

            ''' Iterate over cells '''
            for h in range(a, b, self.cell_size):
                for v in range(c, d, self.cell_size):
                    ''' Iteration over each cell '''
                    features = np.zeros(self.nbins, dtype=np.float64)
                    for x_ in range(h, h + self.cell_size + 1):
                        for y_ in range(v, v + self.cell_size + 1):
                            vote = int(angle[x_, y_] // self.step)
                            features[vote] += mag[x_, y_]
                    norm = np.linalg.norm(features)
                    ''' Do not take into flat regions (with (gx, gy) = 0)'''
                    if norm != 0.0:
                        features /= norm
                    else:
                        features = np.zeros(self.nbins, dtype=np.float64)
                    hog.append(features)
            ''' Concatenation of results '''
            hog = np.array(hog).flatten()
            result.append(hog)

        return np.array(result)
