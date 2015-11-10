from itertools import tee, product
from matplotlib import pyplot as plt
import os
import h5py
import numpy as np
from utils import preprocess
from Classifiers import split

__author__ = 'fabee'


class Stack:
    """
    Class that holds an imaging stack and a voxel size.

    :param stack: python ndarray with dimensions x, y, z, color channels
    :param voxel: tuple of len 3 with side length of voxel

    """

    def __init__(self, stack, voxel):

        self.load(stack)

        if np.any(np.array(voxel) % 2 == 0):
            raise ValueError("Voxel size should be odd.")
        self.voxel = voxel

    def load(self, stackfile):
        if not isinstance(stackfile, str):
            self.X = stackfile
            self.P = np.nan * np.empty(stackfile.shape[:3])
        else:
            with h5py.File(stackfile, 'r') as fid:
                self.stack = np.asarray(fid['stack'])
                X = preprocess(np.asarray(fid['stack']))
                X = np.concatenate((X, 0 * X[..., 0][..., None]), axis=-1).squeeze()
                self.X = X
                self.cells = np.asarray(fid['cells']) if 'cells' in fid else None
                self.P = np.asarray(fid['P']) if 'P' in fid else np.nan * np.empty(X.shape[:3])

    def save(self, stackfile):
        mode = 'r+' if os.path.exists(stackfile) else 'w'
        with h5py.File(stackfile, mode) as fid:
            fid.create_dataset('stack', self.stack.shape, dtype=float, data=self.stack)
            fid.create_dataset('P', self.P.shape, dtype=float, data=self.P)
            if self.cells is not None:
                fid.create_dataset('cells', self.cells.shape, dtype=int, data=self.cells)

    @property
    def half_voxel(self):
        """
        :return: half the voxel size
        """
        vi, vj, vk = self.voxel
        hi, hj, hk = int((vi - 1) / 2), int((vj - 1) / 2), int((vk - 1) / 2)
        return hi, hj, hk

    def generate_training_data(self, cells, negative_balance=5):
        """
        Generate a set of training data.

        :param cells: cell locations as m x 3 array
        :param negative_balance: negative training set will be bigger by that factor (default = 5)
        :return: data and labels (X,y) of training set
        """
        X = self.X

        m_neg = negative_balance * len(cells)
        hi, hj, hk = self.half_voxel
        not_cells = np.c_[np.random.randint(hi, X.shape[0] - hi - 1, size=3 * m_neg),
                          np.random.randint(hj, X.shape[1] - hj - 1, size=3 * m_neg),
                          np.random.randint(hk, X.shape[2] - hk - 1, size=3 * m_neg)].astype(int)

        r = np.sqrt(np.sum([h ** 2 for h in self.half_voxel]))

        D = np.sqrt(np.sum((not_cells[:, None, :] - cells[None, ...]) ** 2, axis=2))
        not_cells = not_cells[np.all(D >= r, axis=1)][:min(m_neg, len(not_cells))]
        print('Using %i of %i negative examples' % (len(not_cells), m_neg))

        X_train = self.extract_patches(np.r_[cells, not_cells])
        y_train = np.ones(len(X_train))
        y_train[len(cells):] = -1

        return X_train, y_train

    def update_P(self, detector, stride=5, n_jobs=1):
        X = self.X
        hi, hj, hk = self.half_voxel
        xgrid = range(hi, X.shape[0] - hi - 1, stride)
        ygrid = range(hj, X.shape[1] - hj - 1, stride)
        zgrid = range(hk, X.shape[2] - hk - 1, stride)

        # raster_pos = np.vstack([e.ravel() for e in np.mgrid[xgrid, ygrid, zgrid]]).T

        m = len(xgrid)*len(ygrid)*len(zgrid) // 2000
        #----------------------------------
        # TODO: Remove this later
        from IPython import embed
        embed()
        exit()
        #----------------------------------

        for idx in tee(product(xgrid, ygrid, zgrid), m):
            pos = list(idx)
            X_all, _ = self.extract_patches(pos)

            prob = detector.positive_prob(X_all, n_jobs=n_jobs)
            # ----------------------------------
            # TODO: Remove this later
            from IPython import embed
            embed()
            exit()
            # ----------------------------------
        self.P[xgrid, ygrid, zgrid] = prob.reshape(self.P[xgrid, ygrid, zgrid].shape)
        return raster_pos, self.P

    def detect_cells(self, detector, stride=5, threshold=0.9, refine=True, n_jobs=1):
        # X = self.stack
        # hi, hj, hk = self.half_voxel
        # xgrid = slice(hi, X.shape[0] - hi - 1, stride)
        # ygrid = slice(hj, X.shape[1] - hj - 1, stride)
        # zgrid = slice(hk, X.shape[2] - hk - 1, stride)
        #
        # raster_pos = np.vstack([e.ravel() for e in np.mgrid[xgrid, ygrid, zgrid]]).T
        #
        # # np.c_[x.ravel(), y.ravel(), z.ravel()].astype(int)
        #
        # # ----------------------------------
        # # TODO: Remove this later
        # from IPython import embed
        # embed()
        # exit()
        # # ----------------------------------
        #
        # X_all, idx = self.extract_patches(raster_pos)
        # raster_pos = raster_pos[idx]
        #
        # # ----------------------------------
        #
        # prob = detector.positive_prob(X_all, n_jobs=n_jobs)
        #
        # raster_pos = raster_pos[prob > threshold]
        #
        # if refine:
        #     sh = int((stride + 1) / 2)
        #     dpos = np.vstack([e.ravel() for e in np.mgrid[-sh:sh + 1, -sh:sh + 1, -sh:sh + 1]]).T
        #
        #     raster_pos = np.vstack([e1 + e2 for e1, e2 in product(raster_pos, dpos)])
        #
        #     X_all, idx = self.extract_patches(raster_pos)
        #     raster_pos = raster_pos[idx]
        #     prob = detector.positive_prob(X_all, n_jobs=n_jobs)
        #
        #     P = np.zeros(self.stack.shape[:3])
        #     P[tuple(zip(*raster_pos))] = prob
        #
        #     raster_pos = raster_pos[prob > threshold]
        #     # prob = prob[prob > threshold]
        #     # local_maxi = peak_local_max(S, indices=False,footprint=np.ones((3, 3,3)), threshold_abs=threshold)
        #     # markers = ndi.label(local_maxi)[0]
        #     raster_pos = peak_local_max(P, footprint=np.ones((5, 5, 5)), threshold_abs=threshold)
        #
        #     # raster_pos = get_concave_components(raster_pos, prob)
        return self.update_P(detector, stride=5, n_jobs=1)

    def extract_patches(self, pixels, channels=slice(0, 2)):
        ret = []
        hi, hj, hk = self.half_voxel
        X = self.X

        M, N, K = X.shape[:3]
        px, py, pz = tuple(map(np.asarray, zip(*pixels)))
        idx = (px >= hi) & (py >= hj) & (pz >= hk) & (px < M - hi) & (py < N - hj) & (pz < K - hk)

        for (i, j, k) in pixels[idx]:
            x = X[i - hi:i + hi + 1, j - hj:j + hj + 1, k - hk:k + hk + 1, channels]
            ret.append(x.mean(axis=3).ravel())
        return np.vstack(ret), idx

    def explore(self, raster_pos, prob=None):
        self._explore_state = dict(positions=raster_pos, y=[], idx=0, probability=prob)
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

        if e.key == "y":
            self._explore_state['y'].append(1)
            print(self._explore_state['y'])
            self._explore_state['idx'] += 1
            if self._explore_state['idx'] >= len(self._explore_state['positions']):
                plt.close(self._explore_state['fig'])
                return
        elif e.key == "n":
            self._explore_state['y'].append(-1)
            print(self._explore_state['y'])
            self._explore_state['idx'] += 1
            if self._explore_state['idx'] >= len(self._explore_state['positions']):
                plt.close(self._explore_state['fig'])
                return
        elif e.key in ['8', '5', '4', '6', '+', '-']:
            d = np.zeros(3)
            if e.key in ['8', '5']:
                d[0] = (-1) ** (e.key == '8')
            elif e.key in ['4', '6']:
                d[1] = (-1) ** (e.key == '4')
            elif e.key in ['+', '-']:
                d[2] = (-1) ** (e.key == '-')
            self._explore_state['positions'][self._explore_state['idx']] += d
        elif e.key == 'q':
            plt.close(self._explore_state['fig'])
        else:
            return

        self._explore_state['ax_stack'].clear()
        self._explore_state['ax_ch1'].clear()
        self._explore_state['ax_ch2'].clear()

        i, j, k = self._explore_state['positions'][self._explore_state['idx']]
        if self._explore_state['probability'] is not None:
            print('P(cell)=%.4g' % self._explore_state['probability'][i, j, k])

        self._explore_state['ax_ch1'].imshow(
            self.X[i - hi:i + hi + 1, j - hj:j + hj + 1, k, 0],
            cmap=plt.cm.gray)
        self._explore_state['ax_ch1'].set_title('Channel 1')
        self._explore_state['ax_ch2'].imshow(
            self.X[i - hi:i + hi + 1, j - hj:j + hj + 1, k, 1],
            cmap=plt.cm.gray)
        self._explore_state['ax_ch2'].set_title('Channel 2')

        self.X[i - hi:i + hi + 1, j - hj:j + hj + 1, k - hk:k + hk + 1, 2] = .2
        self._explore_state['ax_stack'].imshow(self.X[..., k, :], cmap=plt.cm.gray)
        self._explore_state['ax_stack'].plot([j - hj, j + hj], [i + hi, i + hi], '-r')
        self._explore_state['ax_stack'].plot([j - hj, j + hj], [i - hi, i - hi], '-r')
        self._explore_state['ax_stack'].plot([j + hj, j + hj], [i - hi, i + hi], '-r')
        self._explore_state['ax_stack'].plot([j - hj, j - hj], [i - hi, i + hi], '-r')
        self._explore_state['ax_stack'].axis('tight')
        self._explore_state['ax_stack'].set_title(
            'Slice %i' % (k,))

        self._explore_state['fig'].canvas.draw()

        self.X[i - hi:i + hi + 1, j - hj:j + hj + 1, k - hk:k + hk + 1, 2] = 0
