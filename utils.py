import warnings
import numpy as np
from scipy import stats
from scipy.ndimage import convolve1d
from scipy.signal import medfilt, hamming


# class Coord2Pixel:
#     def __init__(self, x_bin_centers, y_bin_centers, z_bin_centers):
#         dx = np.diff(x_bin_centers)
#         assert dx.var() < 1e-20, "Bin center spacing should be equidistant"
#         dx = dx[0]
#         print('Bin distance x', dx)
#         dy = np.diff(y_bin_centers)
#         assert dy.var() < 1e-20, "Bin center spacing should be equidistant"
#         dy = dy[0]
#
#         dz = np.diff(z_bin_centers)
#         assert dz.var() < 1e-20, "Bin center spacing should be equidistant"
#         dz = dz[0]
#         print('Bin distance z', dz)
#
#         self.xedges = np.arange(x_bin_centers[0] - dx / 2, x_bin_centers[-1] + 3 * dx / 2, dx)
#         self.yedges = np.arange(y_bin_centers[0] - dy / 2, y_bin_centers[-1] + 3 * dy / 2, dy)
#         self.zedges = np.arange(z_bin_centers[0] - dz / 2, z_bin_centers[-1] + 3 * dz / 2, dz)
#
#     def __call__(self, X):
#         return np.c_[
#             np.digitize(X[:, 0], self.xedges) - 1,
#             np.digitize(X[:, 1], self.yedges) - 1,
#             np.digitize(X[:, 2], self.zedges) - 1
#         ].astype(int)


def extract_patches(X, pixels, voxel, channels=slice(0, 2)):
    ret = []
    vi, vj, vk = voxel
    assert vi % 2 == 1 and vj % 2 == 1 and vk % 2 == 1, 'Voxels must have odd number of pixels at each side.'
    hi, hj, hk = int((vi - 1) / 2), int((vj - 1) / 2), int((vk - 1) / 2)

    M, N, K = X.shape[:3]
    idx = (pixels[:, 0] >= hi) & (pixels[:, 1] >= hj) & (pixels[:, 2] >= hk) & (pixels[:, 0] < M - hi) & \
          (pixels[:, 1] < N - hj) & (pixels[:, 2] < K - hk)

    for (i, j, k) in pixels[idx]:
        x = X[i - hi:i + hi + 1, j - hj:j + hj + 1, k - hk:k + hk + 1, channels]
        ret.append(x.mean(axis=3).ravel())
    return np.vstack(ret)


def preprocess(X):
    X = (X - X.mean()) / X.std()

    X = medfilt(X, kernel_size=(1, 1, 5, 1, 1))

    lp = np.array(X)
    for i, ws in zip([0, 1, 2], [50, 50, 25]):
        h = hamming(ws)
        h /= h.sum()
        convolve1d(lp, h, axis=i, output=lp)
    X = X - lp

    # return (X - X.min()) / (X.max() - X.min())
    return histeq(X, 500)


def histeq(x, bins=500):
    # get image histogram
    h, edges = np.histogram(x.ravel(), bins)
    cdf = h.cumsum().astype(float)  # cumulative distribution function
    cdf /= cdf[-1]  # normalize
    # use linear interpolation of cdf to find new pixel values
    # out = np.interp(x.ravel(), edges[:-1], cdf)
    target = stats.beta.ppf(cdf, .9, 10)
    out = np.interp(x.ravel(), edges[:-1], target)

    return out.reshape(x.shape)


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


def get_concave_components(pos, f):
    visited = np.zeros(len(f), dtype=bool)
    component_idx = []
    while np.any(~visited):
        vidx = np.where(~visited)[0]
        cur_max = vidx[np.argmax(f[~visited])]
        component_idx.append(cur_max)
        active_set = [cur_max]
        visited[cur_max] = True
        visit_neigbours(active_set, pos, f, visited)

    return pos[component_idx]


def visit_neigbours(active_set, pos, f, visited):
    D = np.sum(np.abs(pos[:, None, :] - pos[active_set][None, ...]), axis=2)
    for f_val, elem in zip(f[active_set], D.T):
        idx = (elem == 1) & ~visited & (f <= f_val)
        visited[idx] = True
        if np.any(idx):
            visit_neigbours(np.where(idx)[0], pos, f, visited)
