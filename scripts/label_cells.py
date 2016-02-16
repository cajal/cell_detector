import argparse
from collections import OrderedDict

import matplotlib.pyplot as plt
import seaborn as sns

from aod_cells.schemata import *

# plot_params = dict(cmap=plt.cm.gray, vmin=0, vmax=1)
plot_params = dict(cmap=plt.cm.gray)
plot_paramsP = dict(cmap=sns.blend_palette(['yellow', 'deeppink'], as_cmap=True), zorder=5)


class CellLabeler:
    def __init__(self, X, cells=None, P=None):
        self.X = X
        self.cells = cells
        self.cell_idx = 0 if cells is not None else None

        self.cut = OrderedDict(zip(['row', 'col', 'depth'], [0, 0, 0]))

        self.P = 0 * self.X
        if P is not None:
            i, j, k = [(i - j + 1) // 2 for i, j in zip(self.X.shape, P.shape)]
            self.P[i:-i, j:-j, k:-k] = P

        fig = plt.figure(facecolor='w')
        gs = plt.GridSpec(3, 5)
        ax = dict()
        ax['depth'] = fig.add_subplot(gs[1:3, :2])
        ax['row'] = fig.add_subplot(gs[0, :2], sharex=ax['depth'])
        ax['col'] = fig.add_subplot(gs[1:3, 2], sharey=ax['depth'])
        ax['3d'] = fig.add_subplot(gs[1:3, 3:], projection='3d')

        self.fig, self.ax = fig, ax
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.replot()
        plt.show()

    def replot(self):
        X0 = self.X
        P0 = np.asarray(self.P)

        P0[P0 < 0.005] = np.nan

        row, col, depth = self.cut.values()
        nr, nc, nd = self.X.shape[:3]
        fig, ax = self.fig, self.ax
        for a in ax.values():
            a.clear()


        color = 'red'
        if self.cells is not None and len(self.cells) > 0:
            out = np.asarray(list(self.cut.values()), dtype=int)
            d = np.sqrt(((self.cells - out) ** 2).sum(axis=1))
            if np.any(d <= 5):
                color = 'dodgerblue'

        ax['row'].imshow(X0[row, :, :].T, **plot_params)
        ax['row'].imshow(P0[row, :, :].T, **plot_paramsP)
        ax['row'].plot([0, nc], [depth, depth], '-', lw=.5, zorder=10, color=color)
        ax['row'].plot([col, col], [0, nd], '-', lw=.5, zorder=10, color=color)
        ax['row'].axis('tight')
        ax['row'].set_aspect('equal')
        ax['row'].axis('off')
        ax['row'].set_xlim((0, nc))
        ax['row'].set_title('col-depth plane')

        ax['col'].imshow(X0[:, col, :], **plot_params)
        ax['col'].imshow(P0[:, col, :], **plot_paramsP)
        ax['col'].plot([depth, depth], [0, nr], '-', lw=.5, zorder=10, color=color)
        ax['col'].plot([0, nd], [row, row], '-', lw=.5, zorder=10, color=color)
        ax['col'].axis('tight')
        ax['col'].set_aspect('equal')
        ax['col'].axis('off')
        ax['col'].set_ylim((0, nr))
        ax['col'].set_title('row-depth plane')

        ax['depth'].imshow(X0[:, :, depth], **plot_params)
        ax['depth'].imshow(P0[:, :, depth], **plot_paramsP)
        ax['depth'].plot([col, col], [0, nr], '-', lw=.5, zorder=10, color=color)
        ax['depth'].plot([0, nc], [row, row], '-', lw=.5, zorder=10, color=color)
        ax['depth'].axis('tight')
        ax['depth'].set_xlim((0, nc))
        ax['depth'].set_ylim((0, nr))
        ax['depth'].set_aspect('equal')
        ax['depth'].axis('off')
        ax['depth'].set_title('row-col plane')

        if self.cells is not None and len(self.cells) > 0:
            c = self.cells
            dz = np.abs(c[:, 2] - out[2]) / 5
            dz = dz * (dz <= 1)

            for cc, alpha in zip(c[dz > 0], 1 - dz[dz > 0]):
                ax['depth'].plot(cc[1], cc[0], 'ok', mfc='dodgerblue', alpha=alpha)

            idx = c[:, 2] == depth
            if np.any(idx):
                ax['depth'].plot(c[idx, 1], c[idx, 0], 'ok', mfc='deeppink', alpha=0.5)

            idx = c[:, 0] == row
            if np.any(idx):
                ax['row'].plot(c[idx, 1], c[idx, 2], 'ok', mfc='deeppink', alpha=0.5)

            idx = c[:, 1] == col
            if np.any(idx):
                ax['col'].plot(c[idx, 2], c[idx, 0], 'ok', mfc='deeppink', alpha=0.5)

            ax['3d'].plot(c[:, 0], c[:, 1], c[:, 2], 'ok', mfc='deeppink')

        ax['3d'].plot([row, row], [0, nc], [depth, depth], '--', lw=2, color=color)
        ax['3d'].plot([row, row], [col, col], [0, nd], '--', lw=2, color=color)
        ax['3d'].plot([0, nr], [col, col], [depth, depth], '--', lw=2, color=color)

        plt.draw()

    def _determine_axes(self, event):
        for k, v in self.ax.items():
            if event.inaxes == v:
                return k

    def on_scroll(self, event):
        what = self._determine_axes(event)
        dimensions = list(self.cut.keys())
        if what in dimensions:
            i = dimensions.index(what)
            k = self.cut[what] + event.step
            k = min(self.X.shape[i], max(k, 0))
            self.cut[what] = k
        self.replot()

    def on_key(self, event):
        if event.key in ['t', 'r', 'e']:
            if event.key == 'e':
                self.cell_idx = max(0, self.cell_idx - 1)
            elif event.key == 't':
                self.cell_idx = min(len(self.cells) - 1, self.cell_idx + 1)
            for k, i in zip(self.cut, self.cells[self.cell_idx, :]):
                self.cut[k] = i
        # if event.key == 's':
        #     fname = input('Please enter filename:')
        #     print('Saving')
        #     self.stack.cells = self.cells
        #     self.stack.save(fname)
        #     self.fig.suptitle('File saved to %s' % (fname,))
        if event.key == 'a':
            new_cell = np.asarray(list(self.cut.values()), dtype=int)
            print('Adding new cell at', new_cell)
            self.cells = np.vstack((self.cells, new_cell))
            self.fig.suptitle('New cell added')
        self.replot()

    def on_press(self, event):
        what = self._determine_axes(event)

        if what == 'depth':
            self.cut['row'], self.cut['col'] = int(event.ydata), int(event.xdata)
        elif what == 'row':
            self.cut['depth'], self.cut['col'] = int(event.ydata), int(event.xdata)
        elif what == 'col':
            self.cut['depth'], self.cut['row'] = int(event.xdata), int(event.ydata)

        if what is not None:
            if event.button == 1:
                new_cell = np.asarray(list(self.cut.values()), dtype=int)
                print('Adding new cell at', new_cell)
                if self.cells is None:
                    self.cells = new_cell[None, :]
                else:
                    self.cells = np.vstack((self.cells, new_cell))

            if event.button == 3:
                out = np.asarray(list(self.cut.values()), dtype=int)
                d = abs(self.cells - out).sum(axis=1)
                self.cells = self.cells[d > 3, :]

        self.replot()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Manually label cells in a stack.')

    # parser.add_argument('file', type=str, help='hdf5 file containing the stack (dims row, col, depth, 1, channels)')
    # parser.add_argument('--probability', type=str, help='numpy file containing the probability map for file')
    #
    # args = parser.parse_args()

    # s = Stack(args.file,
    #           preprocessor=lambda x: average_channels(whiten(unsharp_masking(medianfilter(center(x.squeeze()))))))
    # if args.probability:
    #     P = np.load(args.probability)
    # else:
    #     P = None

    stacks = Stacks().project().fetch.as_dict()
    for i, key in enumerate(stacks):
        print(i, '\t'.join(key.values()))
    key = stacks[int(input('Please select dataset: '))]
    cells = (CellLocations() & key).project().fetch.as_dict()
    if len(cells) > 0:
        for i, ckey in enumerate(cells):
            print(i, '\t'.join(ckey.values()))
        selection = input('Do you want to load a set of locations? [press enter for no] ')
        if len(selection) > 0:
            key = cells[int(selection)]
            cells = (CellLocations() & key).fetch1['cells']
        else:
            cells = None
    prep = list(preprocessors.keys())
    for i, name in enumerate(prep):
        print(i, name)
    key['preprocessing'] = prep[int(input('Please select the preprocessing. '))]
    X = Stacks().load(key)

    labeler = CellLabeler(X, cells)
