from itertools import tee, product
from time import sleep
from matplotlib import pyplot as plt
import os
import h5py
import numpy as np
from utils import preprocess
from Classifiers import split
import theano as th
from scipy.optimize import minimize
import itertools

floatX = th.config.floatX
T = th.tensor
import theano.tensor.nnet.conv3d2d

__author__ = 'fabee'

tensor5 = theano.tensor.TensorType('float64', 5 * [False])




class Stack:
    """
    Class that holds an imaging stack and a voxel size.

    :param stack: python ndarray with dimensions x, y, z, color channels
    :param voxel: tuple of len 3 with side length of voxel

    """

    def __init__(self, stack, voxel, linear_channels=2, quadratic_channels=2, preprocessor=preprocess):

        self.preprocessor = preprocessor
        self.load(stack)

        if np.any(np.array(voxel) % 2 == 0):
            raise ValueError("Voxel size should be odd.")
        self.voxel = voxel
        self.linear_channels = linear_channels
        self.quadratic_channels = quadratic_channels
        flt_width, flt_height, flt_time = self.voxel

        self.U = np.random.randn(quadratic_channels, flt_width, flt_height, flt_time, 1)
        self.W = np.random.randn(linear_channels, flt_width, flt_height, flt_time, 1)

        self.U *= 0
        self.W *= 0
        x = np.asarray(list(itertools.product([-1,1], [-1,1], [-1,1])))
        x[np.abs(x).sum(axis=1) == 1]
        x,y,z = x.T
        self.W[0, 1+x, 1+y, 1+z,0 ] = 1

        self.beta = np.random.randn(linear_channels, quadratic_channels)
        self.b = np.random.randn(linear_channels)

    def load(self, stackfile):
        if not isinstance(stackfile, str):
            self.X = stackfile
            self.P = np.nan * np.empty(stackfile.shape[:3])
        else:
            with h5py.File(stackfile, 'r') as fid:
                self.stack = np.asarray(fid['stack'])

                X = self.preprocessor(np.asarray(fid['stack'])).squeeze()
                # X = np.concatenate((X, 0 * X[..., 0][..., None]), axis=-1).squeeze()
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

    def _build_probability_map(self):

        X = self.X.mean(axis=-1)[None, ..., None]  # batch, x, y, z, channels

        _, in_width, in_height, in_depth, _ = X.shape
        X_ = th.shared(np.require(X, dtype=floatX), borrow=True, name='stack')

        batchsize = 1
        in_channels = 1

        linear_channels, quadratic_channels = self.linear_channels, self.quadratic_channels
        flt_width, flt_height, flt_time = self.voxel

        U_ = tensor5()  # quadratic filter
        W_ = tensor5()  # linear filter
        b_ = T.dvector()  # bias
        beta_ = T.dmatrix()

        linear_filter_ = T.nnet.conv3d2d.conv3d(
            signals=X_.dimshuffle(0, 3, 4, 1, 2),
            filters=W_.dimshuffle(0, 3, 4, 1, 2),
            signals_shape=(batchsize, in_depth, in_channels, in_height, in_width),
            filters_shape=(linear_channels, flt_time, in_channels, flt_height, flt_width),
            border_mode='valid')

        quadr_filter_ = T.nnet.conv3d2d.conv3d(
            signals=X_.dimshuffle(0, 3, 4, 1, 2),
            filters=U_.dimshuffle(0, 3, 4, 1, 2),
            signals_shape=(batchsize, in_depth, in_channels, in_height, in_width),
            filters_shape=(quadratic_channels, flt_time, in_channels, flt_height, flt_width),
            border_mode='valid')
        quadr_filter_ = T.tensordot(quadr_filter_ ** 2, beta_, (2, 1)).dimshuffle(0, 1, 4, 2, 3)

        exponent_ = quadr_filter_ + linear_filter_ + b_.dimshuffle('x', 'x', 0, 'x', 'x')
        p_ = T.exp(exponent_).sum(axis=2).squeeze().T
        p_ = p_ / (1 + p_)  * (1-1e-8) + 1e-8 # apply logistic function to log p_ and add a bit of offset for numerical stability
        # p = th.function([U_, W_, beta_, b_], p)
        return p_, U_, W_, beta_, b_

    def build_crossentropy(self):
        y_shape = tuple(i - j + 1 for i, j in zip(self.X.shape, self.voxel))
        Y = np.zeros(y_shape)
        cell_locs = self.cells - np.array([v // 2 for v in self.voxel])
        i, j, k = cell_locs.T
        Y[i, j, k] = 1
        Y_ = th.shared(np.require(Y, dtype=floatX), borrow=True, name='cells')

        p_, U_, W_, beta_, b_ = self._build_probability_map()

        loglik_ = Y_ * T.log(p_) + (1 - Y_) * T.log(1 - p_)
        cross_entropy_ = -T.mean(loglik_) # + .01*np.abs(U_).mean() + .01*np.abs(W_).mean()
        dcross_entropy_ = T.grad(cross_entropy_, [U_, W_, beta_, b_])

        return th.function([U_, W_, beta_, b_], cross_entropy_), th.function([U_, W_, beta_, b_], dcross_entropy_)

    def visualize(self):
        y_shape = tuple(i - j + 1 for i, j in zip(self.X.shape, self.voxel))
        Y = np.zeros(y_shape)
        cell_locs = self.cells - np.array([v // 2 for v in self.voxel])
        i, j, k = cell_locs.T
        Y[i, j, k] = 1

        p_, U_, W_, beta_, b_ = self._build_probability_map()
        p = th.function([U_, W_, beta_, b_], p_)
        P = p(self.U, self.W, self.beta, self.b)

        i, j, k = [v // 2 for v in self.voxel]

        X = np.asarray(self.X)
        X = np.concatenate([X, 0*X[..., 0][...,None]], axis=-1)
        X[i:-i, j:-j, k:-k, -1] = P
        X[i:-i, j:-j, k:-k, 1] = Y

        fig, ax = plt.subplots(2, 1)
        plt.ion()
        plt.show()
        for z in range(X.shape[2]):
            ax[0].imshow(X[..., z, :], cmap=plt.cm.gray, interpolation='nearest')
            ax[1].imshow(X[..., z, -1], cmap=plt.cm.gray, interpolation='nearest')
            ax[0].axis('tight')
            ax[1].axis('tight')
            plt.draw()
            input()
            ax[0].clear()
            ax[1].clear()



    def fit(self):
        ll, dll = self.build_crossentropy()
        slices, shapes = [], []
        i = 0
        for elem in [self.U, self.W, self.beta, self.b]:
            slices.append(slice(i, i+elem.size))
            shapes.append(elem.shape)
            i += elem.size


        def ravel(*args):
            return np.hstack([e.ravel() for e in args])

        def unravel(x):
            return tuple(x[sl].reshape(sh) for sl, sh in zip(slices, shapes))

        def obj(x):
            return ll(*unravel(x))

        def dobj(x):
            return ravel(*dll(*unravel(x)))

        def callback(x):
            print('Cross entropy:', obj(x))

        x0 = ravel(self.U, self.W, self.beta, self.b)

        opt_results = minimize(obj, x0, jac=dobj, method='BFGS', callback=callback)
        self.U, self.W, self.beta, self.b = unravel(opt_results.x)

if __name__ == "__main__":
    s = Stack('data/sanity.hdf5', (3, 3, 3), preprocessor=lambda x: x, quadratic_channels=1, linear_channels=1)
    # s.fit()
    s.visualize()
