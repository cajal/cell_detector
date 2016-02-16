import warnings
import numpy as np
from scipy import stats
from scipy.ndimage import convolve1d
from scipy.signal import medfilt, hamming
from scipy.ndimage.filters import convolve1d


def medianfilter(X, axis=2):
    ks = [1]*len(X.shape)
    ks[axis] = 5
    return medfilt(X, kernel_size=ks)

def unsharp_masking(X):
    lp = np.array(X)
    for i, ws in zip([0, 1, 2], [50, 50, 25]):
        h = hamming(ws)
        h /= h.sum()
        convolve1d(lp, h, axis=i, output=lp)
    return X - lp

def center(X):
    return (X - X.mean()) / X.std()

def normalize(X):
    return (X - X.min()) / (X.max() - X.min())


def histeq(x, bins=500, alpha=.9, beta=5):
    # get image histogram

    h, edges = np.histogram(x.ravel(), bins)
    cdf = h.cumsum().astype(float)  # cumulative distribution function
    cdf /= cdf[-1]  # normalize
    # use linear interpolation of cdf to find new pixel values
    # out = np.interp(x.ravel(), edges[:-1], cdf)
    target = stats.beta.ppf(cdf, alpha, beta)
    out = np.interp(x.ravel(), edges[:-1], target)
    out -= out.mean()
    return out.reshape(x.shape)


def local_standardize(X, kernelsize=(17, 17, 15)):
    local_sq = X ** 2
    local_mean = np.asarray(X)
    for axis, ks in enumerate(kernelsize):
        # w = np.ones(ks) / ks
        w = np.hamming(ks)
        w /= w.sum()
        local_sq = convolve1d(local_sq, w, axis=axis, mode='reflect')
        local_mean = convolve1d(local_mean, w, axis=axis, mode='reflect')
    return  (X - local_mean)/ np.sqrt(local_sq - local_mean**2)


def whiten(X, n=4, cut_off=.4):
    Y = np.fft.fftn(X)
    wx, wy, wz = X.shape
    W = np.sqrt(np.fft.fftfreq(wx)[:, None, None] ** 2
                + np.fft.fftfreq(wy)[None, :, None] ** 2
                + np.fft.fftfreq(wz)[None, None, :] ** 2)
    F = W * np.exp(- (W / cut_off) ** n)

    X = np.fft.ifftn(Y * F).real
    return X

def average_channels(X):
    if len(X.shape) > 3:
        X = X.mean(axis=-1)
    return X


def compute_crange(K, basefactors=2 ** np.arange(-3, 4.)):
    """
    Estimates a good range for C based on the inverse variance in features space
    of the kernel with kernel matrix K.

    See also:
    Chapelle, O., & Zien, A. (2005). Semi-Supervised Classification by Low Density Separation.

    :param K: kernel matrix
    :param basefactors: factors that get multiplied with the inverse variance to get a good range for C
    :returns: basefactors/estimated variance in feature space

    """
    s2 = np.mean(np.diag(K)) - np.mean(K.ravel())
    if s2 == 0.:
        warnings.warn("Variance in feature space is 0. Using 1!")
    return basefactors / s2
