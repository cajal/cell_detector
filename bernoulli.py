import os
import numpy as np
import theano as th
from matplotlib import pyplot as plt
from scipy.optimize import minimize
import theano as th
from collections import OrderedDict
from scipy.ndimage import convolve1d
floatX = th.config.floatX
T = th.tensor
import theano.tensor.nnet.conv3d2d
from scipy.special import beta
from sklearn.metrics import roc_auc_score


tensor5 = theano.tensor.TensorType('float64', 5 * [False])


class BernoulliProcess:
    def __init__(self, voxel):

        if np.any(np.array(voxel) % 2 == 0):
            raise ValueError("Voxel size should be odd.")
        self.voxel = voxel
        self.parameters = OrderedDict()


    def _build_label_stack(self, X, cell_locations):
        y_shape = tuple(i - j + 1 for i, j in zip(X.shape, self.voxel))
        Y = np.zeros(y_shape)

        cell_locations = cell_locations[np.all(cell_locations < Y.shape, axis=1)
                                        & np.all(cell_locations >= 0, axis=1)]
        cell_locs = cell_locations - np.array([v // 2 for v in self.voxel])

        i, j, k = cell_locs.T
        Y[i, j, k] = 1

        return Y

    def _build_crossentropy(self, X, cell_locations):
        Y = self._build_label_stack(X, cell_locations)
        Y_ = th.shared(np.require(Y, dtype=floatX), borrow=True, name='cells')

        p_, parameters_ = self._build_probability_map(X)

        loglik_ = Y_ * T.log(p_) + (1 - Y_) * T.log(1 - p_)
        cross_entropy_ = -T.mean(loglik_)
        dcross_entropy_ = T.grad(cross_entropy_, parameters_)

        return th.function(parameters_, cross_entropy_), th.function(parameters_, dcross_entropy_)

    def set_parameters(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.parameters:
                self.parameters[k] = v

    def P(self, X):
        p_, params_ = self._build_probability_map(X)
        p = th.function(params_, p_)
        return p(*tuple(self.parameters.values()))

    def auc(self, X, cell_locations, **kwargs):
        return roc_auc_score(self._build_label_stack(X, cell_locations).ravel(), self.P(X).ravel(), **kwargs)

    def visualize(self, X, cell_locations=None):
        y_shape = tuple(i - j + 1 for i, j in zip(X.shape, self.voxel))
        # Y = np.zeros(y_shape)
        # if cell_locations is not None:
        #     cell_locs = cell_locations - np.array([v // 2 for v in self.voxel])
        # i, j, k = cell_locs.T
        # Y[i, j, k] = 1

        p_, parameters_ = self._build_probability_map(X)
        p = th.function(parameters_, p_)
        P = p(*self.parameters.values())

        i, j, k = [v // 2 for v in self.voxel]

        # X[i:-i, j:-j, k:-k, -1] = P
        X0 = 0 * X
        X0[i:-i, j:-j, k:-k] = P
        print(P.min(), P.max())

        fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)
        plt.ion()
        plt.show()
        for z in range(X.shape[2]):
            ax[0].imshow(X[..., z], cmap=plt.cm.gray, interpolation='nearest', )
            ax[1].imshow(X0[..., z], cmap=plt.cm.gray, interpolation='nearest')
            if cell_locations is not None:
                cells = cell_locations[cell_locations[:, 2] == z]

                ax[0].plot(cells[:, 1], cells[:, 0], 'or', mfc='dodgerblue', alpha=.8)
                ax[1].plot(cells[:, 1], cells[:, 0], 'or', mfc='dodgerblue', alpha=.8)
            ax[0].axis('tight')
            ax[1].axis('tight')
            plt.draw()
            input()
            ax[0].clear()
            ax[1].clear()

    def cross_entropy(self, X, cell_locations):
        ce, _ = self._build_crossentropy(X, cell_locations)
        return ce(*self.parameters.values()) / np.log(2)

    def fit(self, X, cell_locations, **options):
        ll, dll = self._build_crossentropy(X, cell_locations)
        p_, params_ = self._build_probability_map(X)
        P = th.function(params_, p_)
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
        # todo find a better way than to box constrain the parameters
        opt_results = minimize(obj, x0, jac=dobj, method='L-BFGS-B', callback=callback,
                               bounds=list(zip(-1000 * np.ones(len(x0)), 1000 * np.ones(len(x0)))),
                               options=options)
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
    def __init__(self, voxel, common_channels=2, linear_channels=2, quadratic_channels=2):
        super(RankDegenerateBernoulliProcess, self).__init__(voxel)

        self.linear_channels = linear_channels
        self.quadratic_channels = quadratic_channels
        self.common_channels = common_channels
        flt_width, flt_height, flt_depth = self.voxel

        # horizontal components of the filters
        self.parameters['u_xy'] = np.random.rand(quadratic_channels, flt_width, flt_height, 1)
        self.parameters['u_xy'] /= self.parameters['u_xy'].size

        # certial components of the filters
        self.parameters['u_z'] = np.random.rand(quadratic_channels, 1, flt_depth, 1)
        self.parameters['u_z'] /= self.parameters['u_z'].size

        # horizontal components of the filters
        self.parameters['w_xy'] = np.random.rand(linear_channels, flt_width, flt_height, 1)
        self.parameters['w_xy'] /= self.parameters['w_xy'].size

        # vertical components of the filters
        self.parameters['w_z'] = np.random.rand(linear_channels, 1, flt_depth, 1)
        self.parameters['w_z'] /= self.parameters['w_z'].size

        self.parameters['beta'] = np.random.randn(common_channels, quadratic_channels)
        self.parameters['gamma'] = np.random.randn(common_channels, linear_channels)
        self.parameters['b'] = np.random.randn(common_channels)

    def _build_separable_convolution(self, no_of_filters, X_, in_shape):
        Vxy_ = T.tensor4(dtype=floatX)  # filters, row, col, channel
        Vz_ = T.tensor4(dtype=floatX)  # quadratic filter

        batchsize, in_channels = 1, 1
        in_width, in_height, in_depth, _ = in_shape
        flt_row, flt_col, flt_depth = self.voxel

        # X is row, col, depth, channel
        xy_ = T.nnet.conv2d(
            # expects (batch size, channels, row, col), transform in to (depth, 1, row, col)
            input=X_.dimshuffle(2, 3, 0, 1),
            # expects nb filters, channels, nb row, nb col
            filters=Vxy_.dimshuffle(0, 3, 1, 2),
            filter_shape=(no_of_filters, in_channels, flt_row, flt_col),
            image_shape=(in_depth, in_channels, in_width, in_height),
            border_mode='valid'
        ).dimshuffle(1, 2, 3, 0)  # the output is shaped (filters, row, col, depth)

        retval_, _ = theano.map(
            lambda v, f:
            T.nnet.conv2d(
                # v is (row, col, depth) and well make it
                # (row, 1, col, depth) = (batch size, stack size, nb row, nb col)
                input=v.dimshuffle(0, 'x', 1, 2),
                # f is (1, flt_depth, in_channels=1) and we'll make it
                # (1, 1, in_channels, flt_depth) =  (nb filters, stack size, nb row, nb col)
                filters=f.dimshuffle('x', 0, 2, 1),  # nb filters, stack size, nb row, nb col
                image_shape=(in_width - flt_row + 1, 1, in_height - flt_col + 1, in_depth),
                filter_shape=(1, 1, in_channels, flt_depth),
                border_mode='valid'
            ).squeeze()
            , sequences=(xy_, Vz_))
        return retval_, (Vxy_, Vz_)

    def _build_exponent(self, X):
        X = X[..., None]  # row, col, depth, channels=1
        X_ = th.shared(np.require(X, dtype=floatX), borrow=True, name='stack')

        linear_channels, quadratic_channels, common_channels = \
            self.linear_channels, self.quadratic_channels, self.common_channels

        quadratic_filter_, (Uxy_, Uz_) = self._build_separable_convolution(quadratic_channels, X_, X.shape)
        linear_filter_, (Wxy_, Wz_) = self._build_separable_convolution(linear_channels, X_, X.shape)

        b_ = T.dvector()  # bias
        beta_ = T.dmatrix()
        gamma_ = T.dmatrix()

        quadr_filter_ = T.tensordot(quadratic_filter_ ** 2, beta_, (0, 1)).dimshuffle(3, 0, 1, 2)
        lin_filter_ = T.tensordot(linear_filter_, gamma_, (0, 1)).dimshuffle(3, 0, 1, 2)

        # Uxy = np.random.randn(quadratic_channels, flt_row, flt_col, in_channels)
        # Uz = np.random.randn(quadratic_channels, 1, flt_depth, in_channels)
        # Wxy = np.random.randn(linear_channels, flt_row, flt_col, in_channels)
        # Wz = np.random.randn(linear_channels, 1, flt_depth, in_channels)
        # beta = np.random.randn(common_channels, quadratic_channels)
        # gamma = np.random.randn(common_channels, linear_channels)
        # b = np.random.randn(linear_channels)

        exponent_ = quadr_filter_ + lin_filter_ + b_.dimshuffle(0, 'x', 'x', 'x')
        return exponent_, (Uxy_, Uz_, Wxy_, Wz_, beta_, gamma_, b_)

    def _build_probability_map(self, X):
        exponent_, params_ = self._build_exponent(X)

        p_ = T.exp(exponent_).sum(axis=0)
        p_ = p_ / (1 + p_) * (
            1 - 2*1e-8) + 1e-8  # apply logistic function to log p_ and add a bit of offset for numerical stability

        return p_, params_


    def __str__(self):
        return """
        Range degenerate Bernoulli process

        quadratic components: %i
        linear components: %i
        common components: %i
        """ % (self.quadratic_channels, self.linear_channels, self.common_channels)

    def __repr__(self):
        return self.__str__()
