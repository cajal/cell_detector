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

