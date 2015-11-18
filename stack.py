from bernoulli import FullBernoulliProcess, RankDegenerateBernoulliProcess
from utils import preprocess
import h5py
import numpy as np
class Stack:
    """
    Class that holds an imaging stack and a voxel size.

    :param stack: python ndarray with dimensions x, y, z, color channels
    :param voxel: tuple of len 3 with side length of voxel

    """

    def __init__(self, stack, preprocessor=preprocess):

        self.preprocessor = preprocessor
        self.load(stack)


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



if __name__ == "__main__":
    # s = Stack('data/smaller.hdf5', preprocessor=lambda x: preprocess(x).mean(axis=-1).squeeze())
    # b = FullBernoulliProcess((9, 9, 7), quadratic_channels=3, linear_channels=3)
    # s = Stack('data/sanity.hdf5', preprocessor=lambda x: x.mean(axis=-1).squeeze())
    # s_test = Stack('data/sanity_test.hdf5', preprocessor=lambda x: x.mean(axis=-1).squeeze())

    # b = RankDegenerateBernoulliProcess( (3,3,3), quadratic_channels=10, linear_channels=10)
    # b.fit(s.X, s.cells)
    # b.visualize(s_test.X, s_test.cells)
    s = Stack('data/smaller.hdf5', preprocessor=lambda x: preprocess(x).mean(axis=-1).squeeze())
    b = RankDegenerateBernoulliProcess( (9,9,7), quadratic_channels=10, linear_channels=10)
    # s = Stack('data/2015-08-25_12-49-41_2015-08-25_13-02-18.h5',
    #           preprocessor=lambda x: preprocess(x).mean(axis=-1).squeeze())
    # b = RankDegenerateBernoulliProcess( (17,17,15), quadratic_channels=10, linear_channels=10)
    b.fit(s.X, s.cells)
    b.visualize(s.X, s.cells)
