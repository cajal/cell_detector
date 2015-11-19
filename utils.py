import warnings
import numpy as np
from scipy import stats
from scipy.ndimage import convolve1d
from scipy.signal import medfilt, hamming

from scipy.ndimage.filters import convolve1d



def preprocess(X):
    X = (X - X.mean()) / X.std()

    X = medfilt(X, kernel_size=(1, 1, 5, 1, 1))

    lp = np.array(X)
    for i, ws in zip([0, 1, 2], [50, 50, 25]):
        h = hamming(ws)
        h /= h.sum()
        convolve1d(lp, h, axis=i, output=lp)
    X = X - lp

    # X =  contrast_normalize(X.squeeze())
    # X = (X - X.min()) / (X.max() - X.min())
    # X -= X.min()
    #return X
    return histeq(X, 500)



def histeq(x, bins=500):
    # get image histogram

    h, edges = np.histogram(x.ravel(), bins)
    cdf = h.cumsum().astype(float)  # cumulative distribution function
    cdf /= cdf[-1]  # normalize
    # use linear interpolation of cdf to find new pixel values
    # out = np.interp(x.ravel(), edges[:-1], cdf)
    target = stats.beta.ppf(cdf, .9, 5)
    out = np.interp(x.ravel(), edges[:-1], target)

    return out.reshape(x.shape)

def contrast_normalize(X, kernelsize=(120,120,30)):
    local_sq = X**2
    local_mean = np.asarray(X)
    for axis, ks in enumerate(kernelsize):
        w = np.ones(ks)/ks
        local_sq = convolve1d(local_sq, w, axis=axis, mode='reflect')
        local_mean = convolve1d(local_mean, w, axis=axis, mode='reflect')
    return X / local_sq



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
