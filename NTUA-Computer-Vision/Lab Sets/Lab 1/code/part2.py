
from cv2 import imread, COLOR_BGR2RGB, COLOR_RGB2GRAY, cvtColor
import numpy as np
import utils
from scipy.signal import fftconvolve as conv2
import matplotlib.pyplot as plt


def harris_stephens(I, sigma=2, rho=2.5, k=0.05, threshold=0.005,
                    box_filters=True, plot=False):
    if box_filters:
        corners = np.zeros(shape=I.shape, dtype=bool)
        if len(I.shape) == 3:
            for level in range(I.shape[2]):
                corners[:, :, level] = utils.hessian_approx_criterion(
                    I[:, :, level], sigma, threshold, plot=plot)
        else:
            corners = utils.hessian_approx_criterion(I, sigma, threshold,
                                                     plot=plot)
    else:
        G_r, r_kern_sze = utils.get_gaussian_filter(rho)
        G_s, s_kern_size = utils.get_gaussian_filter(sigma)
        ''' Calculate Sigma and Rho Image '''
        if len(I.shape) == 3:
            I_sigma = utils.conv3(I, G_s)
            I_rho = utils.conv3(I, G_r)
        else:
            I_sigma = conv2(I, G_s, mode='same')
            I_rho = conv2(I, G_r, mode='same')
        J = utils.structural_tensor(I_sigma, G_r)
        structural_eigs = utils.structural_eigenvalues(*J)
        if plot:
            fig = plt.figure()
            fig.suptitle('Multichannel Structural Eigenvalues for σ = {},'
                         'ρ = {}'.format(sigma, rho))
            ax = plt.subplot(121)
            plt.imshow(structural_eigs[0])
            plt.title('λ+')
            ax = plt.subplot(122)
            plt.imshow(structural_eigs[1])
            plt.title('λ-')
        corners = utils.cornerness_criterion(
            utils.angle_criterion(structural_eigs, k),
            s_kern_size, threshold, sigma=sigma)
    return corners


def single_scale_blobs(I, sigma=2, threshold=0.005, plot=False,
                       box_filters=False):
    G_s, kern_size = utils.get_gaussian_filter(sigma)
    if len(I.shape) == 3:
        I_sigma = utils.conv3(I, G_s)
    else:
        I_sigma = conv2(I, G_s, mode='same')
    if not box_filters:
        hessian = utils.hessian_matrix(I_sigma)
    else:
        Lxx, Lyy, Lxy = utils.hessian_approx(I, sigma)
        hessian = np.array([[Lxx, Lxy], [Lxy, Lyy]])
        if len(I.shape) == 3:
            hessian = hessian.transpose(2, 3, 4, 0, 1)
        else:
            hessian = hessian.transpose(2, 3, 0, 1)
    det = np.linalg.det(hessian)
    blobs = utils.cornerness_criterion(det, kern_size, threshold, sigma=sigma)
    return blobs


def multiscale_blobs(I, sigma_0=2, threshold=0.005, N=4, s=1.5, plot=False,
                     box_filters=False):
    scales = [s**i * sigma_0 for i in range(N)]
    I_gray = utils.brightness(I)
    h, w = I_gray.shape
    blobs = np.empty([h, w, N], dtype=bool)
    LoG_metric = np.empty([h, w, N])
    for i, scale in enumerate(scales):
        ch_blobs = single_scale_blobs(I, sigma=scale, threshold=threshold,
                                      box_filters=box_filters)
        if len(I.shape) == 2:
            blobs[:, :, i] = ch_blobs
        else:
            # get blob if ANY of the channels finds a blob
            blobs[:, :, i] = utils.or_channels(ch_blobs)

        G, _ = utils.get_gaussian_filter(scale)

        ''' Calculate gaussian filtered gray scale image '''
        I_sigma = conv2(I_gray, G, mode='same')

        ''' Calculate normalized LoG '''
        dx, dy = np.gradient(I_sigma)
        dx_dx, dx_dy = np.gradient(dx)
        dy_dx, dy_dy = np.gradient(dy)
        LoG_metric[:, :, i] = scale**2 * np.abs(dx_dx + dy_dy)

    for i, sigma in enumerate(scales):
        if i == 0:
            blobs[:, :, i] = blobs[:, :, i] & \
                (LoG_metric[:, :, i] >= LoG_metric[:, :, i + 1])
        elif i == len(scales) - 1:
            blobs[:, :, i] = blobs[:, :, i] & \
                (LoG_metric[:, :, i] >= LoG_metric[:, :, i - 1])
        else:
            blobs[:, :, i] = blobs[:, :, i] & \
                (LoG_metric[:, :, i] >= LoG_metric[:, :, i - 1]) & \
                (LoG_metric[:, :, i] >= LoG_metric[:, :, i + 1])
    return blobs, scales


def harris_laplacian(I, sigma_0=0, rho_0=2.5, k=0.05, threshold=0.005, s=1.5,
                     N=4, figure=False, box_filters=False,
                     global_maxima=False):
    ''' Create differentiation and integration scales '''
    diff_scales = [s**i * sigma_0 for i in range(N)]
    int_scales = [s**i * rho_0 for i in range(N)]
    corners = np.empty(shape=(I.shape[0], I.shape[1], N), dtype=bool)

    LoG_metric = np.empty(shape=(I.shape[0], I.shape[1], N))

    for i, (sigma, rho) in enumerate(zip(diff_scales, int_scales)):
        ch_corners = harris_stephens(I, sigma, rho, k, threshold,
                                     box_filters=box_filters)
        if len(I.shape) == 3:
            # get corner if ANY of the channels finds a corner
            corners[:, :, i] = utils.or_channels(ch_corners)
            I_gray = utils.brightness(I)
        else:
            corners[:, :, i] = ch_corners
            I_gray = I

        G_s, _ = utils.get_gaussian_filter(sigma)

        ''' Calculate Sigma and Rho Image '''
        I_sigma = conv2(I_gray, G_s, mode='same')

        ''' Calculate normalized LoG '''
        dx, dy = np.gradient(I_sigma)
        dx_dx, dx_dy = np.gradient(dx)
        dy_dx, dy_dy = np.gradient(dy)

        LoG_metric[:, :, i] = sigma**2 * np.abs(dx_dx + dy_dy)

    if global_maxima:
        opt_LoG = np.argmax(LoG_metric, axis=2)

    for i, sigma in enumerate(diff_scales):
        if global_maxima:
            corners[:, :, i] = corners[:, :, i] & (opt_LoG == i)
        else:
            if i == 0:
                corners[:, :, i] = corners[:, :, i] & \
                    (LoG_metric[:, :, i] >= LoG_metric[:, :, i + 1])
            elif i == len(diff_scales) - 1:
                corners[:, :, i] = corners[:, :, i] & \
                    (LoG_metric[:, :, i] >= LoG_metric[:, :, i - 1])
            else:
                corners[:, :, i] = corners[:, :, i] & \
                    (LoG_metric[:, :, i] >= LoG_metric[:, :, i - 1]) & \
                    (LoG_metric[:, :, i] >= LoG_metric[:, :, i + 1])

    return corners, diff_scales


if __name__ == '__main__':
    ''' Parameters '''
    # General parameters
    R = 2.5  # variation of G_r

    # Angles detection
    SIGMA = 2  # variation of G_s
    K = 0.05  # constant for angle criterion
    THETA_CORN = 0.005  # thresholding for cornerness criterion
    s = 1.5  # scale factor
    N = 4  # number of scales

    # Blobs detection
    SIGMA_H = 2
    MULTISCALE_SIGMA_H = SIGMA_H
    THRESHOLD_H = 0.05  # thresholding for cornerness criterion

    # with box filter
    THRESHOLD_BOX_H = THRESHOLD_H

    ''' Read images '''
    blood_smear = cvtColor(imread('images/blood_smear.jpg'), COLOR_BGR2RGB)
    mars = cvtColor(imread('images/mars.png'), COLOR_BGR2RGB)
    urban_edges = cvtColor(imread('images/urban_edges.jpg'), COLOR_BGR2RGB)

    '''
                                Corners detector
    '''

    # single scale
    blood_smear_corners = harris_stephens(blood_smear, SIGMA, R, K, THETA_CORN,
                                         box_filters=False, plot=True)
    mars_corners = harris_stephens(mars, SIGMA, R, K, THETA_CORN,
                                       box_filters=False, plot=True)
    urban_edges = harris_stephens(urban_edges, SIGMA, R, K, THETA_CORN,
                                       box_filters=False, plot=True)
    fig = plt.figure()
    fig.suptitle('Harris-Stephens Corner Detection')
    ax = plt.subplot(121)
    utils.interest_points_visualization(
        blood_smear, utils.or_channels(blood_smear_corners), SIGMA, fig, ax)
    plt.title('Detected corners in blood_smear.jpg')

    ax = plt.subplot(122)
    utils.interest_points_visualization(
        mars, utils.or_channels(mars_corners), SIGMA, fig, ax)
    plt.title('Detected corners in mars.png')

    # ax = plt.subplot(123)
    # utils.interest_points_visualization(
    #     urban_edges, utils.or_channels(urban_edges_corners), SIGMA, fig, ax)
    # plt.title('Detected corners in urban_edges.jpg')

    # multiscale
    mars_multiscale_corners, mars_diff_scales = \
        harris_laplacian(mars, SIGMA, R, K, THETA_CORN, s, N,
                         box_filters=False)
    blood_smear_multiscale_corners, blood_smear_diff_scales = \
        harris_laplacian(blood_smear, SIGMA, R, K, THETA_CORN, s, N,
                         box_filters=False)
    fig = plt.figure()
    fig.suptitle('Harris-Laplacian Corner Detection')
    ax = plt.subplot(121)
    utils.multiscale_interest_points_visualization(
        blood_smear, blood_smear_multiscale_corners,
        blood_smear_diff_scales, fig, ax)
    plt.title('Detected corners in blood_smear.jpg')

    ax = plt.subplot(122)
    utils.multiscale_interest_points_visualization(
        mars, mars_multiscale_corners, mars_diff_scales, fig, ax)
    plt.title('Detected corners in mars.png')

    ''' box filters '''
    # single scale
    blood_smear_corners_bf = harris_stephens(blood_smear, SIGMA, R, K,
                                            THETA_CORN, box_filters=True)
    mars_corners_bf = harris_stephens(mars, SIGMA, R, K, THETA_CORN,
                                          box_filters=True)
    fig = plt.figure()
    fig.suptitle('Harris-Stephens Corner Detection with box filters')

    ax = plt.subplot(121)
    utils.interest_points_visualization(
        blood_smear, utils.or_channels(blood_smear_corners_bf), SIGMA, fig, ax)
    plt.title('Detected corners in blood_smear.jpg with box filters')

    ax = plt.subplot(122)
    utils.interest_points_visualization(
        mars, utils.or_channels(mars_corners_bf), SIGMA, fig, ax)
    plt.title('Detected corners in mars.png with box filters')

    # multiscale
    ''' Multiscale Detection with Box Filters '''
    mars_multiscale_corners_bf, _ = \
        harris_laplacian(mars, SIGMA, R, K, THETA_CORN, s, N,
                         box_filters=True)
    blood_smear_multiscale_corners_bf, blood_smear_diff_scales = \
        harris_laplacian(blood_smear, SIGMA, R, K, THETA_CORN, s, N,
                         box_filters=True)
    fig = plt.figure()
    fig.suptitle('Harris-Laplacian Corner Detection with box filters')

    ax = plt.subplot(121)
    utils.multiscale_interest_points_visualization(
        blood_smear, blood_smear_multiscale_corners_bf,
        blood_smear_diff_scales, fig, ax)
    plt.title('Detected corners in blood_smear.jpg with box filters')

    ax = plt.subplot(122)
    utils.multiscale_interest_points_visualization(
        mars, mars_multiscale_corners_bf,
        mars_diff_scales, fig, ax)
    plt.title('Detected corners in mars.png wit box filters')


    '''
                                Blobs detector
    '''

    # Single scale
    fig = plt.figure()
    fig.suptitle('One scale blob detection')

    blood_smear_blobs = single_scale_blobs(blood_smear, sigma=SIGMA_H,
                                          threshold=THRESHOLD_H)
    ax = plt.subplot(121)
    utils.interest_points_visualization(
        blood_smear, utils.or_channels(blood_smear_blobs), SIGMA_H, fig, ax)
    plt.title('Blobs detected in blood_smear.jpg')

    mars_blobs = single_scale_blobs(
        mars, sigma=SIGMA_H, threshold=THRESHOLD_H)
    ax = plt.subplot(122)
    utils.interest_points_visualization(
        mars, utils.or_channels(mars_blobs), SIGMA_H, fig, ax)
    plt.title('Blobs detected in mars.png')

    # Multiscale
    fig = plt.figure()
    fig.suptitle('Multiscale blob detection (N=4)')

    blood_smear_multiscale_blobs, scales = multiscale_blobs(
        blood_smear, sigma_0=MULTISCALE_SIGMA_H, threshold=THRESHOLD_H)
    ax = plt.subplot(121)
    utils.multiscale_interest_points_visualization(
        blood_smear, blood_smear_multiscale_blobs, scales, fig, ax)
    plt.title('Blobs detected in blood_smear.jpg')

    mars_multiscale_blobs, scales = multiscale_blobs(
        mars, sigma_0=MULTISCALE_SIGMA_H, threshold=THRESHOLD_H)
    ax = plt.subplot(122)
    utils.multiscale_interest_points_visualization(
        mars, mars_multiscale_blobs, scales, fig, ax)
    plt.title('Blobs detected in mars.png')

    ''' Box filters '''

    # Single scale
    fig = plt.figure()
    fig.suptitle('One scale blob detection with box filters')

    blood_smear_blobs = single_scale_blobs(
        blood_smear, sigma=SIGMA_H, threshold=THRESHOLD_BOX_H, box_filters=True)
    ax = plt.subplot(121)
    utils.interest_points_visualization(
        blood_smear, utils.or_channels(blood_smear_blobs), SIGMA_H, fig, ax)
    plt.title('Blobs detected in blood_smear.jpg')

    mars_blobs = single_scale_blobs(
        mars, sigma=SIGMA_H, threshold=THRESHOLD_BOX_H, box_filters=True)
    ax = plt.subplot(122)
    utils.interest_points_visualization(
        mars, utils.or_channels(mars_blobs), SIGMA_H, fig, ax)
    plt.title('Blobs detected in mars.png')

    # Multiscale
    fig = plt.figure()
    fig.suptitle('Multiscale blob detection (N=4) with box filters')

    blood_smear_multiscale_blobs, scales = multiscale_blobs(
        blood_smear, sigma_0=MULTISCALE_SIGMA_H, threshold=THRESHOLD_BOX_H,
        box_filters=True)
    ax = plt.subplot(121)
    utils.multiscale_interest_points_visualization(
        blood_smear, blood_smear_multiscale_blobs, scales, fig, ax)
    plt.title('Blobs detected in blood_smear.jpg')

    mars_multiscale_blobs, scales = multiscale_blobs(
        mars, sigma_0=MULTISCALE_SIGMA_H, threshold=THRESHOLD_BOX_H,
        box_filters=True)
    ax = plt.subplot(122)
    utils.multiscale_interest_points_visualization(
        mars, mars_multiscale_blobs, scales, fig, ax)
    plt.title('Blobs detected in mars.png')

    plt.show()
