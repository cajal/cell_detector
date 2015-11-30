import datajoint as dj
from stack import Stack
from bernoulli import FullBernoulliProcess, RankDegenerateBernoulliProcess
from utils import *
import seaborn as sns
from djaddon import gitlog

schema = dj.schema('datajoint_aod_cell_detection', locals())
import git
import itertools

preprocessors = {
    'center_medianfilter_unsharpmask_histeq':
        lambda x: histeq(unsharp_masking(medianfilter(center(x.squeeze()))), 500).mean(axis=-1),
    'center_medianfilter':
        lambda x: medianfilter(center(x.squeeze())).mean(axis=-1),
    'center_medianfilter_unsharpmask':
        lambda x: unsharp_masking(medianfilter(center(x.squeeze()))).mean(axis=-1),
    'center_medianfilter_unsharpmask_whiten':
        lambda x: whiten(unsharp_masking(medianfilter(center(x.squeeze())))).mean(axis=-1),
}


@schema
class Stacks(dj.Lookup):
    definition = """
    # input training files

    file_name   : varchar(100)  # filename
    vx          : int # x voxel size
    vy          : int # y voxel size
    vz          : int # z voxel size
    ---
    """

    contents = [
        ('data/2015-08-25_12-49-41_2015-08-25_13-02-18.h5', 17, 17, 15),
        ('data/2015-08-25_13-49-54_2015-08-25_13-57-23.h5', 17, 17, 15),
        ('data/2015-08-25_14-36-29_2015-08-25_14-44-41.h5', 17, 17, 15),
    ]


# @schema
# class TrainTestPairs(dj.Lookup):
#     definition = """
#     # input test files
#     ->Stacks
#     test_file_name   : varchar(100)  # filename
#     ---
#     """
#
#     def _prepare(self):
#         self.insert( (Stacks()*Stacks().project(test_file_name='file_name') - 'file_name = test_file_name').fetch(),
#                      skip_duplicates=True)
#

@schema
class ComponentNumbers(dj.Lookup):
    definition = """
    # number of linear and quadratic components for parameter selection

    linear_components          : int # number of linear components
    quadratic_components       : int # number of quadratic components
    common_components          : int
    ---

    """

    @property
    def contents(self):
        yield from itertools.starmap(lambda a, b: a + b,
                                itertools.product(zip(range(2, 12, 2), range(2, 12, 2)), zip(range(2, 12, 2))))

@schema
class Repetitions(dj.Lookup):
    definition = """
    # multiple repetitions for fitting

    idx     : int # repetition index
    ---

    """

    @property
    def contents(self):
        yield from zip(range(10))


@schema
class Preprocessing(dj.Lookup):
    definition = """
    preprocessing       : varchar(200)
    """

    @property
    def contents(self):
        return [(k,) for k in preprocessors]


@schema
@gitlog
class TrainedBSTM(dj.Computed):
    definition = """
    # trained BSTM models

    -> Stacks
    -> ComponentNumbers
    -> Repetitions
    -> Preprocessing
    ---

    u_xy                     : longblob # quadratic filters xy
    u_z                      : longblob # quadratic filters z
    w_xy                     : longblob # linear filters xy
    w_z                      : longblob # linear filters z
    beta                     : longblob # quadratic coefficients
    gamma                    : longblob # linear  coefficients
    b                        : longblob # offsets
    train_cross_entropy      : double   # in Bits/component
    train_auc                : double   # in Bits/component
    train_auc_weighted       : double   # in Bits/component
    """

    def _make_tuples(self, key):
        key_sub = dict(key)
        s = Stack(key['file_name'], preprocessor=preprocessors[key['preprocessing']])
        voxel = (Stacks() & key).fetch1['vx', 'vy', 'vz']
        b = RankDegenerateBernoulliProcess(voxel,
                                           quadratic_channels=key['quadratic_components'],
                                           linear_channels=key['linear_components'],
                                           common_channels=key['common_components']
                                           )
        b.fit(s.X, s.cells, maxiter=100)
        key.update(b.parameters)
        key['train_cross_entropy'] = b.cross_entropy(s.X, s.cells)
        key['train_auc_weighted'] = b.auc(s.X, s.cells, average='weighted')
        key['train_auc'] = b.auc(s.X, s.cells, average='macro')
        self.insert1(key)

    def key2BSTM(self, key):
        trained = (self() & key).fetch1()
        voxel = key['vx'], key['vy'], key['vz']
        b = RankDegenerateBernoulliProcess(voxel, quadratic_channels=key['quadratic_components'],
                                           linear_channels=key['linear_components'],
                                           common_channels=key['common_components'])
        b.set_parameters(**trained)
        return b


@gitlog
@schema
class TestedBSTM(dj.Computed):
    definition = """
    -> TrainedBSTM
    test_file_name          : varchar(100)  # filename
    ---
    test_cross_entropy      : double
    test_auc                : double # ROC area under the curve
    test_auc_weighted       : double # ROC area under the curve weighted by class label imbalance
    """

    @property
    def populated_from(self):
        return TrainedBSTM()*Stacks().project(test_file_name='file_name') - 'file_name = test_file_name'


    def _make_tuples(self, key):
        b = TrainedBSTM().key2BSTM(key)
        s_test = Stack(key['test_file_name'], preprocessor=preprocessors[key['preprocessing']])

        key['test_auc'] = b.auc(s_test.X, s_test.cells, average='macro')
        key['test_auc_weighted'] = b.auc(s_test.X, s_test.cells, average='weighted')
        key['test_cross_entropy'] = b.cross_entropy(s_test.X, s_test.cells)
        self.insert1(key)


if __name__ == "__main__":
    TrainedBSTM().populate(reserve_jobs=True)
    TestedBSTM().populate(reserve_jobs=True)
#     # TrainedBSTM().plot()
#     TestRDBernoulliProcess().populate(reserve_jobs=True)
