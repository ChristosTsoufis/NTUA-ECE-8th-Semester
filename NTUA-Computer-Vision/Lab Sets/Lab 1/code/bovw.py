
from sklearn.svm import LinearSVC
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import part2
import utils
import argparse
import cv2
import numpy as np
import glob
import os
import pickle


class BOVW:
    def __init__(self, directory, descriptor_name, detector_name,
                 n_clusters=50):
        ''' Initialize BoVW Classifier
            Args:
            directory: Dataset directory
            descriptor_name: Name of descriptor
            n_clusters: Number of clusters
        '''
        self.directory = directory
        self.n_clusters = n_clusters
        self.descriptor_name = descriptor_name
        self.detector_name = detector_name
        if self.descriptor_name == 'sift':
            self.descriptor = cv2.xfeatures2d.SIFT_create()
        elif self.descriptor_name == 'hog':
            self.descriptor = utils.HOG()
        elif self.descriptor_name == 'surf':
            self.descriptor = cv2.xfeatures2d.SURF_create()
        else:
            raise NotImplementedError('Descriptor not available')

        self.label_encoder = preprocessing.LabelEncoder()

        self.clf = LinearSVC()

    def extract_features(self, lfeatures=None, sfeatures=None):
        ''' Extract features using the descriptor provided in constructor'''
        print('Extracting Features')
        cwd = os.getcwd()
        if lfeatures is None:
            self.labels = os.listdir(self.directory)
            os.chdir(self.directory)
            images, self.X, Y = [], [], []
            for label in self.labels:
                img_list = os.listdir(label)
                for img_file in img_list:
                    Y.append(label)
                    print('Loading Image: ' + img_file)
                    I = cv2.imread(os.path.join(label, img_file))
                    kp = self.detect(I)

                    if self.descriptor_name in ['sift', 'surf']:
                        _, des = self.descriptor.compute(I, kp)
                    elif self.descriptor_name == 'hog':
                        des = self.descriptor.compute(I, kp)

                    print(des.shape)
                    self.X.append(des)

            os.chdir('..')
            self.label_encoder.fit(self.labels)

            self.Y = self.label_encoder.transform(Y)
        else:
            data = utils.load_pickle(lfeatures)
            self.X = data['X']
            self.Y = data['Y']
            self.labels = data['labels']
            self.directory = lfeatures

        os.chdir(cwd)

        ''' Train Test Split '''
        self.X_train, self.X_test, self.Y_train, self.Y_test = \
            train_test_split(self.X, self.Y, test_size=0.33, random_state=42)

        if sfeatures is not None:
            with open(sfeatures, 'wb+') as f:
                data = {
                    'X': self.X,
                    'Y': self.Y,
                    'labels': self.labels
                }
                utils.dump(data, f)

    def extract_embeddings(self, p=0.5):
        ''' Extracts embeddings for the visual words '''
        def histc(u):
            ''' Extracts Embeddings (Histogram of Visual Words) '''
            emb = np.zeros(shape=(len(u), self.n_clusters), dtype=np.float64)
            for i in range(len(u)):
                predictions = self.kmeans.predict(u[i])
                for prediction in predictions:
                    emb[i, prediction] += 1
                ''' Normalize with L2 Norm '''
                emb[i] /= np.linalg.norm(emb[i])
            return emb

        train_size = int(p * sum(len(u) for u in self.X))
        print('Number of samples for kmeans training: {}'.format(train_size))
        ''' Stack Everything '''

        X_train_stacked = np.vstack(self.X_train)
        np.random.shuffle(X_train_stacked)
        X_train_kmeans = X_train_stacked[:train_size]

        ''' Train KMeans Classifier '''
        self.kmeans = KMeans(
            n_clusters=self.n_clusters, n_init=self.n_clusters // 10,
            random_state=0)
        print('Fitting KMeans Classifier')
        self.kmeans.fit(X_train_kmeans)

        ''' Construct embeddings vectors '''
        self.X_train_emb = histc(self.X_train)
        self.X_test_emb = histc(self.X_test)

    def train(self):
        ''' Trains SVM Classifier '''
        self.clf.fit(self.X_train_emb, self.Y_train)

    def evaluate(self):
        ''' Evaluation of the classifier '''
        y_ = self.clf.predict(self.X_test_emb)

        accuracy = np.sum(y_ == self.Y_test) / len(y_)
        print('Accuracy: {}%'.format(round(accuracy * 100, 2)))
        print('Classification Reports')
        print(classification_report(self.Y_test, y_, target_names=self.labels))


    def detect(self, I):
        SIGMA = 2
        SIGMA_BOX = 4
        R = 2.5
        K = 0.05
        THETA_CORN = 0.005
        s = 1.5
        N = 2
        if self.detector_name == 'harris_laplacian':
            corners, scales = \
                part2.harris_laplacian(I, SIGMA, R, K, THETA_CORN, s, N,
                                       global_maxima=True, box_filters=False)
            return utils.features_to_keypoints(corners, scales)
        elif self.detector_name == 'harris_stephens':
            corners = \
                part2.harris_stephens(I, SIGMA, R, K, THETA_CORN,
                                      box_filters=False)
            return utils.features_to_keypoints(utils.or_channels(corners), SIGMA)
        elif self.detector_name == 'single_scale_blobs':
            blobs = part2.single_scale_blobs(I, sigma=SIGMA)
            return utils.features_to_keypoints(utils.or_channels(blobs), SIGMA)
        elif self.detector_name == 'multiscale_blobs':
            blobs, scales = part2.multiscale_blobs(I, N=2, sigma_0=SIGMA)
            return utils.features_to_keypoints(blobs, scales)
        else:
            raise NotImplementedError('Unsupported ROI detector')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        usage='Classifier using BoVW Model and SVM')
    argparser.add_argument('-n', help='Number of clusters',
                           default=50, type=int)
    argparser.add_argument('-p', help='Percentage of datapoints for KMeans',
                           default=0.5, type=float)
    argparser.add_argument('--descriptor', help='Descriptor (sift, hog, surf)',
                           default='sift')
    argparser.add_argument('--detector',
                           help='Detector (harris_laplacian, blobs)',
                           default='harris_laplacian')
    argparser.add_argument(
        '-i', help='Input directory',
        default='cv21_lab1_part3_material/classification/Data')
    argparser.add_argument('--save',
                           action='store_true',
                           help='Dump pickle with object')
    argparser.add_argument('--lfeatures', help='Load features from pickle',
                           default=None)
    argparser.add_argument('--sfeatures', help='Save features to pickle',
                           default=None)

    args = argparser.parse_args()

    bovw = BOVW(args.i, args.descriptor, args.detector, args.n)

    bovw.extract_features(args.lfeatures, args.sfeatures)
    print('Extracting Embeddings')
    bovw.extract_embeddings(args.p)
    print('Training')
    bovw.train()
    print('Evaluation')
    bovw.evaluate()

    if args.save:
        utils.dump(bovw, 'bovw_{}_{}_{}.p'.format(args.n, args.descriptor,
                                           args.detector))
