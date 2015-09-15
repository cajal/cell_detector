import argparse
import pickle
import numpy as np
import h5py
from train import Stack, Detector
from utils import preprocess
from scipy import io


def parse_command_line():
    help = """
    Detect cells in a stack and save the cell positions.
    """

    parser = argparse.ArgumentParser(description=help)
    parser.add_argument('stackfile',
                        help="""hdf5 files file containing the stack (dataset stack and cells). The file should contain
                        a dataset called 'stack'.""")
    parser.add_argument('detector', help='Trained detector generated with train.py.')
    parser.add_argument('outfile', help='Matlab file storing the pixel positions in the variable "cells"')
    parser.add_argument('--prob', '-q', help='Positive probability threshold (default 0.95) ',
                        default=0.95, type=float)
    parser.add_argument('--stride', '-s', help='Stride when searching for new cells (default 5).',
                        default=5, type=int)

    return parser.parse_args()


if __name__ == '__main__':
    # --- command line parsing
    args = parse_command_line()

    # --- load data
    with h5py.File(args.stackfile, 'r') as fid:
        X = preprocess(np.asarray(fid['stack'])).squeeze()

    with open(args.detector, 'rb') as fid:
        det = pickle.load(fid)

    stk = Stack(X, det.voxel)

    cells, prob = stk.detect_cells(det, args.stride, args.prob)
    print('Found %i cells' % (len(cells)))
    io.savemat(args.outfile, {'cells': cells})
