import numpy as np
import itertools
import h5py

f = h5py.File('data/sanity.hdf5')
X = np.asarray(f['stack'])*0
f.close()

cells = []
x = np.asarray(list(itertools.product(*(3*[[-1,0,1]]))))
x = x[np.abs(x).sum(axis=1) == 1]
x,y,z = x.T
for i,j,k in zip(np.random.randint(2, 47, 200), np.random.randint(2,44,200), np.random.randint(2,27,200)):
    X[i+x, j+y, k+z,0,:] = 1
    cells.append((i,j,k))

cells = np.asarray(cells, dtype=int)
N = np.random.rand(*X.shape[:-1])[...,None]
N = np.concatenate([N,N], axis=4)
X[N < 0.005] = 1

f = h5py.File('data/sanity.hdf5','w')
f.create_dataset('cells', cells.shape, dtype=int, data=cells)
f.create_dataset('stack', X.shape, dtype=float, data=X)
f.close()
