from schemata import *
import argparse
import h5py
import matplotlib.pyplot as plt
import seaborn as sns


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
    parser.add_argument('--plot', type=str, help='plot detected cells to this path')

    return parser.parse_args()


def max_iterator(P, n, voxel):
    P = np.asarray(P)
    i, j, k = [i // 2 for i in voxel]
    counter = 0

    while counter < n:
        cell = np.where(P == P.max())
        x, y, z = cell[0][0], cell[1][0], cell[2][0]
        P[x - i:x + i, y - j:y + j, z - k:z + k] = -1
        yield (x, y, z)
        counter += 1


if __name__ == '__main__':
    # --- command line parsing
    args = parse_command_line()

    key = (ValidationBSTM() * StackGroup().aggregate(ValidationBSTM(),
                                                     max_auc='MAX(val_auc_weighted)') & 'val_auc_weighted=max_auc' & dict(
        group_name=args.stacktype)).fetch1()
    voxel = key['vx'], key['vy'], key['vz']

    preprocessor = preprocessors[key['preprocessing']]
    model = TrainedBSTM().key2BSTM(key)

    with h5py.File(args.stackfile) as fid:
        X = preprocessor(np.asarray(fid['stack'], dtype=float).squeeze())

    P = model.P(X, full=True)

    cells = list(max_iterator(P, args.sites, voxel))

    if args.plot:
        sns.set_style('white')

        for i, (x, y, z) in enumerate(cells):
            fig, ax = plt.subplots()
            ax.imshow(X[..., z], cmap=plt.cm.gray, interpolation='nearest')
            ax.plot(y, x, 'ok', mfc='orange')
            ax.set_aspect(1)
            ax.axis('tight')
            fig.savefig('%s/cell%03i.png' % (args.plot, i))
            plt.close(fig)

    with h5py.File(args.outfile,'w') as fid:
        cells = np.asarray(cells, dtype=int)
        fid.create_dataset('cells', cells.shape, dtype=int, data=cells+1)
        fid.create_dataset('P', P.shape, dtype=float, data=P)



