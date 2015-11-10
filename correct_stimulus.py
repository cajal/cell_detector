import argparse
from matplotlib import pyplot as plt
from scipy import io
import numpy as np

from stack import Stack
from utils import preprocess
import h5py


def parse_command_line():
    help = """
    Correct cell positions in a labeled dataset and store in a h5 file.
    """

    parser = argparse.ArgumentParser(description=help)
    parser.add_argument('stackfile', help='matlab file containing the stack')
    parser.add_argument('positions', help='Positions used for training.')
    parser.add_argument('outfile', help='hdf5 outputfile.')
    parser.add_argument('--voxel', '-v',
                        help='voxel size (comma separated list). Voxels size lengths must be odd. (default 17,17,13)',
                        default='19,19,15')

    return parser.parse_args()


def load_data(args, process=True):
    voldata = io.loadmat(args.stackfile)['voldata'].astype(float)  # data/2015-08-25_12-49-41_2015-08-25_13-02-18.mat
    if args.positions:
        p = io.loadmat(args.positions)['p'].astype(int) - 1  # data/cell_locations.mat
    else:
        p = None

    if process:
        voldata = preprocess(voldata.astype(float))
        X = np.concatenate((voldata, 0 * voldata[..., 0][..., None]), axis=-1).squeeze()
    else:
        X = voldata
    return X, p


if __name__ == '__main__':
    # --- command line parsing
    args = parse_command_line()

    X, p = load_data(args)
    X0, _ = load_data(args, process=False)

    voxel = tuple(int(e) for e in args.voxel.split(','))
    for v in voxel:
        assert v % 2 == 1, 'Voxel side lengths must be odd'
    stk = Stack(X, voxel)
    stk.explore(p.copy())


    with h5py.File(args.outfile, 'w') as fid:
        fid.create_dataset('stack', X0.shape, dtype=float, data=X0)
        fid.create_dataset('cells', p.shape, dtype=int, data=stk._explore_state['positions'])
