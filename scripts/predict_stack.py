import argparse

import h5py
import matplotlib.pyplot as plt
import seaborn as sns

from aod_cells.schemata import *


def parse_command_line():
    help = """
    Correct cell positions in a labeled dataset and store in a h5 file.
    """

    parser = argparse.ArgumentParser(description=help)
    groups = ', '.join(StackGroup().fetch['group_name'])
    parser.add_argument('stacktype', help='current choices are: ' + groups)
    parser.add_argument('stackfile', help='hdf5 file containing the 250x250x60 stack as "stack"')
    parser.add_argument('outfile', help='Output file hdf5 file. The probability is stored as "P" the cells as "cells"')
    parser.add_argument('--sites', type=int, help='number of sites to find (default=10)', default=10)
    parser.add_argument('--correct', type=int,
                        help='''correct cell locations in a radius of n around the old
                                locations supplied as "cells" in the hdf5 file''')
    parser.add_argument('--plot', type=str, help='plot detected cells to this path')

    return parser.parse_args()


def max_iterator(P, n, voxel):
    P = np.array(P) # copy
    i, j, k = [i // 2 for i in voxel]
    counter = 0

    while counter < n:
        cell = np.where(P == P.max())
        x, y, z = cell[0][0], cell[1][0], cell[2][0]
        P[x - i:x + i, y - j:y + j, z - k:z + k] = -1
        yield (x, y, z)
        counter += 1

def correction_iterator(P, old_locations, radius, voxel):

    offset = np.asarray([i // 2 for i in voxel], dtype=int)
    for loc in old_locations:
        fro = np.asarray([max(0, e) for e in loc - radius - offset])
        to = np.asarray([min(b, e) for b, e in zip(P.shape, loc + radius + offset)])
        p = P[tuple(slice(f,t) for f,t in zip(fro, to))]
        locs = np.mgrid[tuple(slice(f,t) for f,t in zip(fro, to))]
        yield locs[:, p == p.max()]



if __name__ == '__main__':
    # --- command line parsing
    args = parse_command_line()

    key = (ValidationBSTM() * StackGroup().aggregate(
                ValidationBSTM() & 'linear_components <= 4' & 'quadratic_components <= 4' & 'common_components <= 4',
                max_auc='MAX(val_auc_weighted)') & 'val_auc_weighted=max_auc' & dict(group_name=args.stacktype)).fetch1()

    voxel = key['vx'], key['vy'], key['vz']

    preprocessor = preprocessors[key['preprocessing']]
    model = TrainedBSTM().key2BSTM(key)

    with h5py.File(args.stackfile) as fid:
        X = preprocessor(np.asarray(fid['stack'], dtype=float).squeeze())
        # subtract 1 for matlab python index compatibility
        old_cells = np.asarray(fid['cells'], dtype=int) - 1 if 'cells' in fid else None


    P = model.P(X, full=True) # get probability map

    with h5py.File(args.outfile) as fid:
        if args.correct:
            assert old_cells is not None, "You must specify cell locations in 'cells' for correction"
            new_cells = list(correction_iterator(P, old_cells, args.correct, voxel))
            new_cells = np.asarray(new_cells, dtype=int)
            fid.create_dataset('corrected_cells', new_cells.shape, dtype=int, data=new_cells + 1)
        else:
            new_cells = list(max_iterator(P, args.sites, voxel))
            new_cells = np.asarray(new_cells, dtype=int)
            fid.create_dataset('cells', new_cells.shape, dtype=int, data=new_cells + 1)
            fid.create_dataset('P', P.shape, dtype=float, data=P)

    if args.plot:
        sns.set_style('white')

        for i, (x, y, z) in enumerate(new_cells):
            fig, ax = plt.subplots()
            ax.imshow(X[..., z], cmap=plt.cm.gray, interpolation='nearest')
            ax.plot(y, x, 'ok', mfc='orange')
            ax.set_aspect(1)
            ax.axis('tight')
            fig.savefig('%s/cell%03i.png' % (args.plot, i))
            plt.close(fig)
