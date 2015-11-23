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
    'center_medianfilter': lambda x: medianfilter(center(x.squeeze())).mean(axis=-1),
    'center_medianfilter_unsharpmask': lambda x: unsharp_masking(medianfilter(center(x.squeeze()))).mean(axis=-1),
    'center_medianfilter_unsharpmask_whiten': lambda x: whiten(unsharp_masking(medianfilter(center(x.squeeze())))).mean(axis=-1),
}


@schema
class TrainingFiles(dj.Lookup):
    definition = """
    # input training files

    file_name   : varchar(100)  # filename
    ---
    vx          : int # x voxel size
    vy          : int # y voxel size
    vz          : int # z voxel size
    """

    contents = [
        ('data/2015-08-25_12-49-41_2015-08-25_13-02-18.h5', 17, 17, 15),
    ]


@schema
class TestingFiles(dj.Lookup):
    definition = """
    # input testing files

    test_file_name   : varchar(100)  # filename
    vx          : int # x voxel size
    vy          : int # y voxel size
    vz          : int # z voxel size
    """

    contents = [
        ('data/2015-08-25_13-49-54_2015-08-25_13-57-23.h5', 17, 17, 15),
    ]


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
        yield from zip(range(5, 45, 5), range(5, 45, 5), range(5,45,5))


@schema
class Repetitions(dj.Lookup):
    definition = """
    # multiple repetitions for fitting
    
    idx     : int # repetition index
    ---
    
    """

    @property
    def contents(self):
        yield from zip(range(5))


@schema
class Preprocessing(dj.Lookup):
    definition = """
    preprocessing       : varchar(200)
    """

    @property
    def contents(self):
        return [(k, ) for k in preprocessors]

@gitlog
@schema
class TrainedRDBernoulliProcess(dj.Computed):
    definition = """
    # Trained ranl degenerate Bernoulli Processes

    -> TrainingFiles
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
    """

    def _make_tuples(self, key):
        key_sub = dict(key)
        s = Stack(key['file_name'], preprocessor=preprocessors[key['preprocessing']])
        voxel = (TrainingFiles() & key).fetch1['vx', 'vy', 'vz']
        b = RankDegenerateBernoulliProcess(voxel,
                                           quadratic_channels=key['quadratic_components'],
                                           linear_channels=key['linear_components'],
                                           common_channels=key['common_components']
                                           )
        b.fit(s.X, s.cells, maxiter=100)
        key.update(b.parameters)
        key['train_cross_entropy'] = b.cross_entropy(s.X, s.cells)
        self.insert1(key)

@gitlog
@schema
class TestRDBernoulliProcess(dj.Computed):
    definition = """
    -> TrainedRDBernoulliProcess
    -> TestingFiles
    ---
    test_cross_entropy      : double
    test_auc                : double # ROC area under the curve
    """

    @property
    def populated_from(self):
        return TrainedRDBernoulliProcess() * TestingFiles() * TrainingFiles() & TrainedRDBernoulliProcess()

    def _make_tuples(self, key):

        if key['file_name'] != key['test_file_name']:
            trained = (TrainedRDBernoulliProcess() & key).fetch1()
            voxel = key['vx'], key['vy'], key['vz']
            b = RankDegenerateBernoulliProcess(voxel, quadratic_channels=key['quadratic_components'],
                                                      linear_channels=key['linear_components'],
                                                      common_channels=key['common_components'])
            b.set_parameters(**trained)

            s = Stack(key['test_file_name'], preprocessor=preprocessors[key['preprocessing']])

            key['test_auc'] = b.auc(s.X, s.cells)
            key['test_cross_entropy'] = b.cross_entropy(s.X, s.cells)
            self.insert1(key)

if __name__ == "__main__":
    # TrainedRDBernoulliProcess().populate(reserve_jobs=True)
    # TrainedRDBernoulliProcess().plot()
    TestRDBernoulliProcess().populate(reserve_jobs=False)