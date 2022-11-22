
''' Driver file of 3.1. '''
import argparse
from matplotlib import pyplot as plt
import scipy.io as sio
import cv2
import part2
import numpy as np
import utils
import math
import pickle
from PIL import Image


class FeaturesMatcher:
    def __init__(self, descriptor_name, detector_name, matcher):
        self.descriptor_name = descriptor_name
        self.detector_name = detector_name
        self.matcher = self.get_matcher(matcher)
        if descriptor_name == 'sift':
            self.descriptor = cv2.xfeatures2d.SIFT_create()
        elif descriptor_name == 'hog':
            self.descriptor = utils.HOG()
        elif descriptor_name == 'surf':
            self.descriptor = cv2.xfeatures2d.SURF_create()
        else:
            raise NotImplementedError('Descriptor not available')
        if detector_name == 'harris_laplacian':
            self.detector = part2.harris_laplacian
            self.multiscale = True
        elif detector_name == 'harris_stephens':
            self.detector = part2.harris_stephens
            self.multiscale = False
        elif detector_name == 'single_scale_blobs':
            self.detector = part2.single_scale_blobs
            self.multiscale = False
        elif detector_name == 'multiscale_blobs':
            self.detector = part2.multiscale_blobs
            self.multiscale = True

    def match(self, input_file, visualize=False, box_filters=False, **kwargs):
        '''
            Args:
                input_file: Input file should be a .mat with (at least) the
                following:
                    scale_original (scales)
                    theta_original (rotations)

        '''
        lower_limit = kwargs.pop('lower_limit', 2)
        l_state = kwargs.pop('l_state', None)
        s_state = kwargs.pop('s_state', None)
        mat = sio.loadmat(input_file)
        img_set = mat['ImgSet'][0]
        categories = img_set.shape[0]
        total_scale_errors = np.empty(categories)
        total_theta_errors = np.empty(categories)
        for c_indx, category in enumerate(img_set):
            scales = mat['scale_original'][0]
            rotations = mat['theta_original'][0]
            if l_state is None:
                category_images = category[0]
                descriptors = []
                keypoints = []
                scales_h = []
                rotations_h = []
                images = []
                for index, img in enumerate(category_images):
                    I = img.astype(np.uint8)
                    kp = self.detect(I, box_filters=box_filters)
                    keypoints.append(kp)
                    images.append(I)
                    if self.descriptor_name in ['sift', 'surf']:
                        _, des = self.descriptor.compute(I, kp)
                    elif self.descriptor_name == 'hog':
                        des = self.descriptor.compute(I, kp)
                    if scales[index] == 1 and rotations[index] == 0:
                        gold_index = index
                        gold_image = I
                    descriptors.append(des)
                if s_state is not None:
                    data = scales, rotations, descriptors, keypoints, \
                        scales_h, rotations_h, images, gold_index
                    utils.dump(data, s_state + '{}'.format(c_indx))
            else:
                data = utils.load_pickle(l_state + '{}'.format(c_indx))
                scales, rotations, descriptors, keypoints, scales_h, \
                    rotations_h, images, gold_index = data

            print('Found descriptors for all images in category')
            gold_descriptor = descriptors[gold_index]
            for index, descriptor in enumerate(descriptors):
                if index == gold_index:
                    pass
                else:
                    matches = self.get_matches(gold_descriptor, descriptor)
                    if len(matches) < lower_limit:
                        raise ValueError('Uncorrelated images')
                    else:
                        gold_points = np.float32([keypoints[gold_index][m.queryIdx].pt for m in matches])
                        key_points = np.float32(
                            [keypoints[index][m.trainIdx].pt
                                for m in matches])
                        img_points = (np.array([gold_image[i[0], i[1]] for i in gold_points.astype(np.int)])/255).astype(np.float32)
                        obj_points = (np.array([images[index][i[0], i[1]] for i in key_points.astype(np.int)])).astype(np.float32)
                        M, mask = cv2.findHomography(gold_points, key_points,
                                                    cv2.RANSAC, 5)
                        if M is None:
                            raise ValueError('Could not find homography'
                                             ' matrix. Make sure that the'
                                             ' images are correlated and/or'
                                             ' adjust the threshold.')
                        # scale, theta = self.get_scale_theta_from_homography(M)
                        scale = self.get_scale(gold_points, key_points)
                        theta = self.get_theta_from_homography(M)
                        scales_h.append(scale)
                        rotations_h.append(theta)
                        if visualize:
                            self.visualize(images[gold_index], images[index],
                                           keypoints[gold_index],
                                           keypoints[index], matches, M, mask)
            scales = list(scales)
            rotations = list(rotations)
            del scales[gold_index]
            del rotations[gold_index]
            scales = np.array(scales)
            rotations = np.array(rotations)
            scales_d = np.abs(scales - scales_h)
            rotations_d = np.abs(rotations - rotations_h)
            mean_scales_d = np.average(scales_d)
            mean_rotations_d = np.average(rotations_d)
            total_scale_errors[c_indx] = mean_scales_d
            total_theta_errors[c_indx] = mean_rotations_d
            print('Scales mean error for image: {}'.format(mean_scales_d))
            print('Rotations mean error for image: {}'.format(mean_rotations_d))
        print('Total scale errors: {}'.format(np.average(total_scale_errors)))
        print('Total theta errors: {}'.format(np.average(total_theta_errors)))

    def get_scale(self, img_points, obj_points):
        img_points = img_points[0:4]
        obj_points = obj_points[0:4]
        T = cv2.getPerspectiveTransform(obj_points, img_points)
        a = T[0][0]
        b = T[0][1]
        c = T[1][0]
        d = T[1][1]
        sX = math.sqrt(a**2 + b**2)
        sY = math.sqrt(c**2 + d**2)
        return min(sX, sY)

    def detect(self, I, **kwargs):
        box_filters = kwargs.pop('box_filters', False)
        sigma = kwargs.pop('sigma', 2)
        sigma_box = kwargs.pop('sigma_box', 4)
        R = kwargs.pop('R', 2.5)
        K = kwargs.pop('K', 0.05)
        threshold = kwargs.pop('threshold', 0.005)
        s = kwargs.pop('s', 1.5)
        N = kwargs.pop('N', 2)
        if self.multiscale:
            ROI, scales = self.detector(I, sigma_0=sigma, N=N, s=s,
                                        threshold=threshold,
                                        box_filters=box_filters)
            return utils.features_to_keypoints(ROI, scales)
        else:
            ROI = self.detector(I, sigma=sigma, threshold=threshold,
                                box_filters=box_filters)
            return utils.features_to_keypoints(ROI, sigma)

    def get_matcher(self, name):
        if name == 'flann':
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=15)
            search_params = dict(checks=50)
            matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            raise NotImplementedError('This matcher is not yet supported')
        return matcher

    def get_matches(self, descriptor1, descriptor2, **kwargs):
        ''' Get matches based on the matcher and the descriptors and
        filter out outliers '''
        k = kwargs.pop('k', 2)
        threshold = kwargs.pop('threshold', 0.9)
        matches = self.matcher.knnMatch(descriptor1, descriptor2, k=k)
        inliers = []
        for m, n in matches:
            if m.distance < threshold * n.distance:
                inliers.append(m)
        return inliers

    def get_theta_from_homography(self, M):
        K = np.array([[706.4034, 0, 277.2018], [0, 707.9991, 250.6182], [0, 0, 1]])
        ns, Rs, Ts, _ = cv2.decomposeHomographyMat(M, K)
        theta = math.inf
        for solution in Rs:
            r32 = solution[2, 1]
            r33 = solution[2, 2]
            r31 = solution[2, 0]
            thetax = math.atan2(r32, r33)
            thetay = math.atan2(-r31, math.sqrt(r32*r32 + r33*r33))
            if abs(thetax - thetay) < theta:
                theta = (thetax + thetay) / 2
        return theta

    def visualize(self, img1, img2, kp1, kp2, matches, M, mask):
        obj_corners = np.empty((4,1,2), dtype=np.float32)
        obj_corners[0,0,0] = 0
        obj_corners[0,0,1] = 0
        obj_corners[1,0,0] = img1.shape[1]
        obj_corners[1,0,1] = 0
        obj_corners[2,0,0] = img1.shape[1]
        obj_corners[2,0,1] = img1.shape[0]
        obj_corners[3,0,0] = 0
        obj_corners[3,0,1] = img1.shape[0]
        matchesMask = mask.ravel().tolist()
        h, w = img1.shape
        dst = cv2.perspectiveTransform(obj_corners, M)
        img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 1, cv2.LINE_AA)
        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=None,
                           matchesMask=matchesMask,
                           flags=2)
        img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, **draw_params).get()
        plt.imshow(img3, cmap=plt.get_cmap('gray'))
        plt.show()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        usage='Feature Matching')
    argparser.add_argument('--descriptor', help='Descriptor (sift, hog, surf)',
                           default='sift')
    argparser.add_argument('--detector',
                           help='Detector (harris_laplacian, blobs)',
                           default='harris_laplacian')
    argparser.add_argument('--box_filters',
                           help='Use box filters',
                           action='store_true')
    argparser.add_argument(
        '-i', help='Input file (mat)',
        default='cv21_lab1_part3_material/matching/snrImgSet.mat')
    argparser.add_argument('--matcher',
                           help='Matcher for different descriptors',
                           default='flann')
    argparser.add_argument('--s_state', help='Save state',
                           default=None)
    argparser.add_argument('--l_state', help='Load state',
                           default=None)
    argparser.add_argument('--visualize', help='Visualize matchings',
                          action='store_true')
    args = argparser.parse_args()

    matcher = FeaturesMatcher(args.descriptor, args.detector, args.matcher)
    matcher.match(args.i, visualize=args.visualize, l_state=args.l_state,
                  box_filters=args.box_filters, s_state=args.s_state)
