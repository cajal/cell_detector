import os
import numpy as np
import theano as th
from matplotlib import pyplot as plt
from scipy.optimize import minimize
import theano as th
from collections import OrderedDict

floatX = th.config.floatX
T = th.tensor
import theano.tensor.nnet.conv3d2d

tensor5 = theano.tensor.TensorType('float64', 5 * [False])


class BernoulliProcess:
    def __init__(self, voxel):

        if np.any(np.array(voxel) % 2 == 0):
            raise ValueError("Voxel size should be odd.")
        self.voxel = voxel
        self.parameters = OrderedDict()

    def _build_crossentropy(self, X, cell_locations):
        y_shape = tuple(i - j + 1 for i, j in zip(X.shape, self.voxel))
        Y = np.zeros(y_shape)
        cell_locs = cell_locations - np.array([v // 2 for v in self.voxel])
        i, j, k = cell_locs.T
        Y[i, j, k] = 1
        Y_ = th.shared(np.require(Y, dtype=floatX), borrow=True, name='cells')

        p_, parameters_ = self._build_probability_map(X)

        loglik_ = Y_ * T.log(p_) + (1 - Y_) * T.log(1 - p_)
        cross_entropy_ = -T.mean(loglik_)
        dcross_entropy_ = T.grad(cross_entropy_, parameters_)

        return th.function(parameters_, cross_entropy_), th.function(parameters_, dcross_entropy_)

    def visualize(self, X, cell_locations=None):
        y_shape = tuple(i - j + 1 for i, j in zip(X.shape, self.voxel))
        Y = np.zeros(y_shape)
        if cell_locations is not None:
            cell_locs = cell_locations - np.array([v // 2 for v in self.voxel])
        i, j, k = cell_locs.T
        Y[i, j, k] = 1

        p_, parameters_ = self._build_probability_map(X)
        p = th.function(parameters_, p_)
        P = p(*self.parameters.values())

        i, j, k = [v // 2 for v in self.voxel]

        X = np.stack([X, X, 0 * X], axis=3)

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

    def cross_entropy(self, X, cell_locations):
        ce, _ = self._build_crossentropy(X, cell_locations)
        return ce(*self.parameters.values()) / np.log(2)

    def fit(self, X, cell_locations):
        ll, dll = self._build_crossentropy(X, cell_locations)
        slices, shapes = [], []
        i = 0
        for elem in self.parameters.values():
            slices.append(slice(i, i + elem.size))
            shapes.append(elem.shape)
            i += elem.size

        def ravel(params):
            return np.hstack([e.ravel() for e in params])

        def unravel(x):
            return tuple(x[sl].reshape(sh) for sl, sh in zip(slices, shapes))

        def obj(x):
            return ll(*unravel(x))

        def dobj(x):
            return ravel(dll(*unravel(x)))

        def callback(x):
            print('Cross entropy:', obj(x))

        x0 = ravel(self.parameters.values())

        opt_results = minimize(obj, x0, jac=dobj, method='BFGS', callback=callback)
        for k, param in zip(self.parameters, unravel(opt_results.x)):
            self.parameters[k] = param


class FullBernoulliProcess(BernoulliProcess):
    def __init__(self, voxel, linear_channels=2, quadratic_channels=2):
        super(FullBernoulliProcess, self).__init__(voxel)

        self.linear_channels = linear_channels
        self.quadratic_channels = quadratic_channels
        flt_width, flt_height, flt_depth = self.voxel

        self.parameters['u'] = np.random.rand(quadratic_channels, flt_width, flt_height, flt_depth, 1)
        self.parameters['u'] /= self.parameters['u'].size
        self.parameters['w'] = np.random.rand(linear_channels, flt_width, flt_height, flt_depth, 1)
        self.parameters['w'] /= self.parameters['w'].size

        self.parameters['beta'] = np.random.randn(linear_channels, quadratic_channels)
        self.parameters['b'] = np.random.randn(linear_channels)

    def _build_probability_map(self, X):
        X = X[None, ..., None]  # batch, x, y, z, channels
        _, in_width, in_height, in_depth, _ = X.shape
        X_ = th.shared(np.require(X, dtype=floatX), borrow=True, name='stack')

        batchsize, in_channels = 1, 1
        linear_channels, quadratic_channels = self.linear_channels, self.quadratic_channels
        flt_width, flt_height, flt_depth = self.voxel

        U_ = tensor5()  # quadratic filter
        W_ = tensor5()  # linear filter
        b_ = T.dvector()  # bias
        beta_ = T.dmatrix()

        linear_filter_ = T.nnet.conv3d2d.conv3d(
            signals=X_.dimshuffle(0, 3, 4, 2, 1),
            filters=W_.dimshuffle(0, 3, 4, 2, 1),
            signals_shape=(batchsize, in_depth, in_channels, in_height, in_width),
            filters_shape=(linear_channels, flt_depth, in_channels, flt_height, flt_width),
            border_mode='valid')

        quadr_filter_ = T.nnet.conv3d2d.conv3d(
            signals=X_.dimshuffle(0, 3, 4, 2, 1),
            filters=U_.dimshuffle(0, 3, 4, 2, 1),
            signals_shape=(batchsize, in_depth, in_channels, in_height, in_width),
            filters_shape=(quadratic_channels, flt_depth, in_channels, flt_height, flt_width),
            border_mode='valid')

        quadr_filter_ = T.tensordot(quadr_filter_ ** 2, beta_, (2, 1)).dimshuffle(0, 1, 4, 2, 3)

        exponent_ = quadr_filter_ + linear_filter_ + b_.dimshuffle('x', 'x', 0, 'x', 'x')
        p_ = T.exp(exponent_).sum(axis=2).squeeze().T
        p_ = p_ / (1 + p_) * (
            1 - 1e-8) + 1e-8  # apply logistic function to log p_ and add a bit of offset for numerical stability
        return p_, (U_, W_, beta_, b_)


class RankDegenerateBernoulliProcess(BernoulliProcess):
    def __init__(self, voxel, linear_channels=2, quadratic_channels=2):
        super(RankDegenerateBernoulliProcess, self).__init__(voxel)

        self.linear_channels = linear_channels
        self.quadratic_channels = quadratic_channels
        flt_width, flt_height, flt_depth = self.voxel

        self.parameters['Uxy'] = np.random.rand(quadratic_channels, flt_width, flt_height)
        self.parameters['Uxy'] /= self.parameters['Uxy'].size

        self.parameters['Uz'] = np.random.rand(quadratic_channels, flt_depth)
        self.parameters['Uz'] /= self.parameters['Uz'].size

        self.parameters['Wxy'] = np.random.rand(linear_channels, flt_width, flt_height)
        self.parameters['Wxy'] /= self.parameters['Wxy'].size

        self.parameters['Wz'] = np.random.rand(linear_channels, flt_depth)
        self.parameters['Wz'] /= self.parameters['Wz'].size

        self.parameters['beta'] = np.random.randn(linear_channels, quadratic_channels)
        self.parameters['b'] = np.random.randn(linear_channels)

    def _build_probability_map(self, X):
        # ----------------------------------
        # TODO: Remove this later
        from IPython import embed
        embed()
        exit()
        # ----------------------------------


        # input: (batch size, channels, rows, columns)
        # filters: (number of filters, channels, rows, columns)

        X = X[None, ..., None]  # batch, x, y, z, channels
        _, in_width, in_height, in_depth, _ = X.shape
        X_ = th.shared(np.require(X, dtype=floatX), borrow=True, name='stack')

        batchsize, in_channels = 1, 1
        linear_channels, quadratic_channels = self.linear_channels, self.quadratic_channels
        flt_width, flt_height, flt_depth = self.voxel

        U_ = tensor5()  # quadratic filter
        W_ = tensor5()  # linear filter
        b_ = T.dvector()  # bias
        beta_ = T.dmatrix()

        linear_filter_ = T.nnet.conv3d2d.conv3d(
            signals=X_.dimshuffle(0, 3, 4, 2, 1),
            filters=W_.dimshuffle(0, 3, 4, 2, 1),
            signals_shape=(batchsize, in_depth, in_channels, in_height, in_width),
            filters_shape=(linear_channels, flt_depth, in_channels, flt_height, flt_width),
            border_mode='valid')

        quadr_filter_ = T.nnet.conv3d2d.conv3d(
            signals=X_.dimshuffle(0, 3, 4, 2, 1),
            filters=U_.dimshuffle(0, 3, 4, 2, 1),
            signals_shape=(batchsize, in_depth, in_channels, in_height, in_width),
            filters_shape=(quadratic_channels, flt_depth, in_channels, flt_height, flt_width),
            border_mode='valid')

        quadr_filter_ = T.tensordot(quadr_filter_ ** 2, beta_, (2, 1)).dimshuffle(0, 1, 4, 2, 3)

        exponent_ = quadr_filter_ + linear_filter_ + b_.dimshuffle('x', 'x', 0, 'x', 'x')
        p_ = T.exp(exponent_).sum(axis=2).squeeze().T
        p_ = p_ / (1 + p_) * (
            1 - 1e-8) + 1e-8  # apply logistic function to log p_ and add a bit of offset for numerical stability
        return p_, (U_, W_, beta_, b_)
