from itertools import product
import pickle
from pprint import pprint
from matplotlib import pyplot as plt
from scipy import io
import numpy as np
from scipy.spatial.distance import pdist
from sklearn.grid_search import GridSearchCV
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.svm import SVC
from scipy import ndimage as ndi

from skimage.morphology import watershed
from skimage.feature import peak_local_max

from utils import preprocess, compute_crange, extract_patches, get_concave_components
import argparse


class Detector:
    def __init__(self, voxel):
        self._training_data = None
        self.voxel = voxel

    @property
    def training_data(self):
        self._training_data = (X_train, y_train)

    def add_training_data(self, X_train, y_train):
        self._training_data = (np.vstack((self._training_data[0], X_train)),
                               np.vstack((self._training_data[1], y_train)))

    def fit(self, X_train, y_train):
        self._training_data = X_train, y_train

        gamma = 1 / np.median(pdist(X_train, 'euclidean'))
        C = compute_crange(rbf_kernel(X_train, gamma=gamma), basefactors=2 ** np.arange(-3, 6.))

        # Set the parameters by cross-validation
        parameter_grid = [{'kernel': ['rbf'], 'gamma': list(2 ** np.arange(-3, 6.) * gamma), 'C': list(C)}]

        pprint("Considering the following parameters")
        pprint(parameter_grid)

        clf = GridSearchCV(SVC(C=1, probability=True), parameter_grid, cv=5, scoring='accuracy', n_jobs=10,
                           verbose=True)
        clf.fit(X_train, y_train)

        pprint("Found best estimator")
        pprint(clf.best_estimator_)
        pprint(clf.best_score_)

        self.classifier = clf.best_estimator_

    def retrain(self):
        self.fit(*self.training_data)

    def positive_prob(self, X):
        prob_all = self.classifier.predict_proba(X)
        return prob_all[:, 1]


class Stack:
    def __init__(self, stack, voxel):
        self.stack = stack
        self.voxel = voxel

    @property
    def half_voxel(self):
        vi, vj, vk = self.voxel
        hi, hj, hk = int((vi - 1) / 2), int((vj - 1) / 2), int((vk - 1) / 2)
        return hi, hj, hk

    def generate_training_data(self, cells, negative_balance=5):
        X = self.stack

        m_neg = negative_balance * len(cells)
        hi, hj, hk = self.half_voxel
        not_cells = np.c_[np.random.randint(hi, X.shape[0] - hi - 1, size=3 * m_neg),
                          np.random.randint(hj, X.shape[1] - hj - 1, size=3 * m_neg),
                          np.random.randint(hk, X.shape[2] - hk - 1, size=3 * m_neg)].astype(int)

        idx = np.ones(len(not_cells), dtype=bool)
        r = np.sqrt(np.sum([h ** 2 for h in self.half_voxel]))

        D = np.sqrt(np.sum((not_cells[:, None, :] - cells[None, ...]) ** 2, axis=2))
        not_cells = not_cells[np.all(D >= r, axis=1)][:min(m_neg, len(not_cells))]
        print('Using %i of %i negative examples' % (len(not_cells), m_neg))

        X_train = extract_patches(X, np.r_[cells, not_cells], self.voxel)
        y_train = np.ones(len(X_train))
        y_train[len(cells):] = -1

        return X_train, y_train

    def detect_cells(self, detector, stride=5, threshold=0.9, refine=True):
        X = self.stack
        hi, hj, hk = self.half_voxel
        # x, y, z = np.meshgrid(range(hi, X.shape[0] - hi, stride), range(hj, X.shape[1] - hj, stride),
        #                       range(hk, X.shape[2] - hk, stride))
        #
        raster_pos = np.vstack([e.ravel() for e in np.mgrid[hi:X.shape[0] - hi + 1:stride,
                                                   hj:X.shape[1] - hj + 1:stride,
                                                   hk:X.shape[2] - hk + 1:stride]]).T

        # np.c_[x.ravel(), y.ravel(), z.ravel()].astype(int)

        X_all = extract_patches(self.stack, raster_pos, self.voxel)
        # ----------------------------------

        prob = detector.positive_prob(X_all)

        raster_pos = raster_pos[prob > threshold]

        if refine:
            sh = int((stride + 1) / 2)
            dpos = np.vstack([e.ravel() for e in np.mgrid[-sh:sh + 1, -sh:sh + 1, -sh:sh + 1]]).T
            raster_pos = np.vstack([e1 + e2 for e1, e2 in product(raster_pos, dpos)])
            X_all = extract_patches(self.stack, raster_pos, self.voxel)
            prob = detector.positive_prob(X_all)
            raster_pos = raster_pos[prob > threshold]
            prob = prob[prob > threshold]
            S = np.zeros(self.stack.shape[:3])
            S[tuple(zip(*raster_pos))] = prob
            # local_maxi = peak_local_max(S, indices=False,footprint=np.ones((3, 3,3)), threshold_abs=threshold)
            # markers = ndi.label(local_maxi)[0]
            raster_pos = peak_local_max(S, footprint=np.ones((3, 3, 3)), threshold_abs=threshold)


            # raster_pos = get_concave_components(raster_pos, prob)
        return raster_pos

    def explore(self, raster_pos):
        self._explore_state = dict(positions=raster_pos, y=[], idx=0,
                                   z=raster_pos[0, 2])
        gs = plt.GridSpec(2, 3)
        fig = plt.figure()
        self._explore_state['ax_stack'] = fig.add_subplot(gs[:2, :2])
        self._explore_state['ax_ch1'] = fig.add_subplot(gs[0, 2])
        self._explore_state['ax_ch2'] = fig.add_subplot(gs[1, 2])
        self._explore_state['fig'] = fig
        fig.canvas.mpl_connect('key_press_event', self)
        plt.show()

    def __call__(self, e):
        vi, vj, vk = self.voxel
        hi, hj, hk = int((vi - 1) / 2), int((vj - 1) / 2), int((vk - 1) / 2)

        if e.key == "down":
            self._explore_state['z'] = min(self._explore_state['z'] + 1, self.stack.shape[2] - 1)
        elif e.key == "up":
            self._explore_state['z'] = max(self._explore_state['z'] - 1, 0)
        elif e.key == "y":
            self._explore_state['y'].append(1)
            print(self._explore_state['y'])
            self._explore_state['idx'] += 1
            if self._explore_state['idx'] >= len(self._explore_state['positions']):
                plt.close(self._explore_state['fig'])
                return
            self._explore_state['z'] = self._explore_state['positions'][self._explore_state['idx'], 2]
        elif e.key == "n":
            self._explore_state['y'].append(-1)
            print(self._explore_state['y'])
            self._explore_state['idx'] += 1
            if self._explore_state['idx'] >= len(self._explore_state['positions']):
                plt.close(self._explore_state['fig'])
                return
            self._explore_state['z'] = self._explore_state['positions'][self._explore_state['idx'], 2]
        elif e.key == 'q':
            plt.close(self._explore_state['fig'])
        else:
            return

        self._explore_state['ax_stack'].clear()
        self._explore_state['ax_ch1'].clear()
        self._explore_state['ax_ch2'].clear()

        i, j, k = self._explore_state['positions'][self._explore_state['idx']]

        self._explore_state['ax_ch1'].imshow(
            self.stack[i - hi:i + hi + 1, j - hj:j + hj + 1, self._explore_state['z'], 0],
            cmap=plt.cm.gray)
        self._explore_state['ax_ch1'].set_title('Channel 1')
        self._explore_state['ax_ch2'].imshow(
            self.stack[i - hi:i + hi + 1, j - hj:j + hj + 1, self._explore_state['z'], 1],
            cmap=plt.cm.gray)
        self._explore_state['ax_ch2'].set_title('Channel 2')

        self.stack[i - hi:i + hi + 1, j - hj:j + hj + 1, k - hk:k + hk + 1, 2] = .3
        self._explore_state['ax_stack'].imshow(self.stack[..., self._explore_state['z'], :], cmap=plt.cm.gray)
        self._explore_state['ax_stack'].set_title(
            'Slice %i (%i)' % (self._explore_state['z'], self._explore_state['z'] - k))
        self._explore_state['fig'].canvas.draw()
        self.stack[i - hi:i + hi + 1, j - hj:j + hj + 1, k - hk:k + hk + 1, 2] = 0


def parse_command_line():
    help = """
    (Re)Train a cell detector on a recorded stack and identified positions.
    """

    parser = argparse.ArgumentParser(description=help)
    parser.add_argument('stackfile', help='matlab file containing the stack')
    parser.add_argument('--voxel', '-v',
                        help='voxel size (comma separated list). Voxels size lengths must be odd. (default 17,17,13)',
                        default='17,17,13')
    parser.add_argument('--stride', '-s', help='Stride when searching for new cells (default 5).',
                        default=5, type=int)
    parser.add_argument('--mnegative', '-m', help='How many times more negative examples do you want (default 5).',
                        default=5, type=int)
    parser.add_argument('--detector', '-d', help='Use previously trained detector in that file.')
    parser.add_argument('--positions', '-p', help='Positions used for training.')
    parser.add_argument('--outfile', '-o', help='Outfile, where the detector will be stored (defailt detector.pkl). ',
                        default='detector.pkl')
    parser.add_argument('--prob', '-q', help='Positive probability threshold (default 0.9) ',
                        default=0.9, type=float)

    return parser.parse_args()


def load_data(args):
    voldata = io.loadmat(args.stackfile)['voldata']  # data/2015-08-25_12-49-41_2015-08-25_13-02-18.mat
    if args.positions:
        p = io.loadmat(args.positions)['p'].astype(int) - 1  # data/cell_locations.mat
    else:
        p = None

    voldata = preprocess(voldata.astype(float))
    X = np.concatenate((voldata, 0 * voldata[..., 0][..., None]), axis=-1).squeeze()
    return X, p


if __name__ == '__main__':
    # --- command line parsing
    args = parse_command_line()

    # --- load data
    X, p = load_data(args)
    voxel = tuple(int(e) for e in args.voxel.split(','))
    for v in voxel:
        assert v % 2 == 1, 'Voxel side lengths must be odd'
    stk = Stack(X, voxel)


    # --- set voxel size and stride

    if args.detector:  # if detector is specified
        with open(args.detector, 'rb') as fid:
            det = pickle.load(fid)
        stk.voxel = det.voxel
        cells = stk.detect_cells(det, args.stride, args.prob)
        print('Found %i cells' % (len(cells)))
        stk.explore(cells)


    elif p is not None:  # if not, train it
        X_train, y_train = stk.generate_training_data(p)

        det = Detector(voxel)
        det.fit(X_train, y_train)
        with open(args.outfile, 'wb') as fid:
            pickle.dump(det, fid, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        pass
