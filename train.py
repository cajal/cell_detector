import pickle
import argparse

import h5py
import numpy as np

from Classifiers import Detector
from stack import Stack
from utils import preprocess


def parse_command_line():
    help = """
    (Re)Train a cell detector on a recorded stack and identified positions.

    If a detector is supplied via -d, the script assumes that the detector should be retrained. In that case,
    the detector is run on the current   stack, extracts all cells and shows them one by one to be reclassified.
    The window for reclassification can be controlled as follows: adjust window (8, 4, 5, 6 i.e. directions on num block),
    adjust depth (+, -), classify as cell (y), classify as not-cell (n), quit and retrain (q). The retrained classifier
    will be stored in the argument of -o (or detector.pkl if not supplied). If no detector is supplied and if the hdf5
    file contains a dataset called 'cells' which specifies the cell positions in pixels, then a new detector is trained
    and stored in the file supplied by -o.
    """

    parser = argparse.ArgumentParser(description=help)
    parser.add_argument('stackfile',
                        help='hdf5 files file containing the stack and the cell positions (dataset stack and cells)')
    parser.add_argument('--voxel', '-v',
                        help='voxel size (comma separated list). Voxels size lengths must be odd. (default 17,17,13)',
                        default='17,17,13')
    parser.add_argument('--stride', '-s', help='Stride when searching for new cells (default 5).',
                        default=5, type=int)
    parser.add_argument('--mnegative', '-m', help='How many times more negative examples do you want (default 5).',
                        default=5, type=int)
    parser.add_argument('--detector', '-d', help='Use previously trained detector in that file.')
    parser.add_argument('--outfile', '-o', help='Outfile, where the detector will be stored (defailt detector.pkl). ',
                        default='detector.pkl')
    parser.add_argument('--prob', '-q', help='Positive probability threshold (default 0.9) ',
                        default=0.9, type=float)

    return parser.parse_args()


if __name__ == '__main__':
    # --- command line parsing
    args = parse_command_line()

    # --- load data
    with h5py.File(args.stackfile, 'r') as fid:
        X = preprocess(np.asarray(fid['stack']))
        X = np.concatenate((X, 0 * X[..., 0][..., None]), axis=-1).squeeze()
        if 'cells' in fid:
            p = np.asarray(fid['cells'])
        else:
            p = None

    voxel = tuple(int(e) for e in args.voxel.split(','))
    for v in voxel:
        assert v % 2 == 1, 'Voxel side lengths must be odd'
    stk = Stack(X, voxel)


    # --- set voxel size and stride
    if args.detector:  # if detector is specified
        with open(args.detector, 'rb') as fid:
            det = pickle.load(fid)
        stk.voxel = det.voxel
        cells, prob = stk.detect_cells(det, args.stride, args.prob)
        print('Found %i cells' % (len(cells)))
        stk.explore(cells, prob=prob)

        p = stk._explore_state['positions']
        y = np.asarray(stk._explore_state['y'])
        with h5py.File(args.stackfile, 'r+') as fid:
            if not 'cells' in fid:
                print('Adding positions to hdf5 file')
                fid.create_dataset('cells', p.shape, dtype=int, data=p)

        Xnew, idx = stk.extract_patches(p)
        p = p[idx]
        det.add_training_data(Xnew, y)
        det.retrain()
        with open(args.outfile, 'wb') as fid:
            pickle.dump(det, fid, protocol=pickle.HIGHEST_PROTOCOL)

    elif p is not None:  # if not, train it
        X_train, y_train = stk.generate_training_data(p)

        det = Detector(voxel)
        det.fit(X_train, y_train)
        with open(args.outfile, 'wb') as fid:
            pickle.dump(det, fid, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        pass
