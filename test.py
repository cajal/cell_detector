from matplotlib import pyplot as plt
from scipy import io
import numpy as np
from train import Stack

from utils import preprocess

locals().update(io.loadmat('data/2015-08-25_12-49-41_2015-08-25_13-02-18.mat'))
locals().update(io.loadmat('data/cell_locations.mat'))
voldata = preprocess(voldata.astype(float))

p = p.astype(int)


# voldata[..., 0] = histeq(voldata[..., 0])
# voldata[..., 1] = histeq(voldata[..., 1])
# voldata = histeq(voldata)


X = np.concatenate((voldata, 0 * voldata[..., 0][..., None]), axis=-1).squeeze()


stk = Stack(X, (19,19,15))

stk.explore(p-1)

# fig, ax = plt.subplots()
# plt.ion()
# plt.show()
#
# for (i, j, k) in p:
#     X[i - 8:i + 9, j - 8:j + 9, k - 6:k + 7, 2] = .3
#
# for i in range(X.shape[2]):
#     ax.clear()
#     ax.imshow(X[..., i, :], cmap=plt.cm.jet)
#     # if np.any(p[:,2] == i):
#     #     # idx = (p[:,2] > i - 5) & (p[:,2] < i + 5)
#     #     idx = p[:,2] == i
#     #     x = p[idx]
#     #     ax.plot(x[:,1], x[:,0],'ok', color='orange')
#     ax.axis('tight')
#     plt.draw()
#
#     input()
