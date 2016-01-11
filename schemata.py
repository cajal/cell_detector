import datajoint as dj
from stack import Stack
from bernoulli import FullBernoulliProcess, RankDegenerateBernoulliProcess
from utils import *
import seaborn as sns
from djaddon import gitlog, hdf5
import h5py
from itertools import repeat
import itertools
from djaddon.slack import notify_user

APITOKEN = open('token.txt').read().strip()
schema = dj.schema('datajoint_cell_detection', locals())
# schema = dj.schema('fabee_cell_detection', locals())

preprocessors = {
    'histogram_equalization':
        lambda x: histeq(unsharp_masking(medianfilter(center(average_channels(x)))), 500),
    # 'center_medianfilter':
    #     lambda x: medianfilter(center(average_channels(x))),
    'basic':
        lambda x: unsharp_masking(medianfilter(center(average_channels(x)))),
    'whitening':
        lambda x: whiten(unsharp_masking(medianfilter(center(average_channels(x))))),
    'normalize':
        lambda x: local_standardize(medianfilter(average_channels(x))),
}

groups = dict(
    manolis082015=[
        ('data/2015-08-25_12-49-41_2015-08-25_13-02-18.h5',),
        ('data/2015-08-25_13-49-54_2015-08-25_13-57-23.h5',),
        ('data/2015-08-25_14-36-29_2015-08-25_14-44-41.h5',),
    ],
    twitch102015=[
        ('data/2015-10-08_17-30-30.h5',),
        ('data/2015-10-08_17-42-24.h5',),
    ],
    jake=[
        ('data/m6252Astack_002.h5',),
    ],
    sr101=[
        ('data/2016-01-07_16-07-01_SR101.h5',)
    ]
)


@schema
class StackGroup(dj.Lookup):
    definition = """
    # input training files
    group_name          : varchar(100) # group name
    ---
    """

    @property
    def contents(self):
        yield from zip(groups)


@schema
class Stacks(dj.Manual):
    definition = """
    ->StackGroup
    file_name   : varchar(100)  # filename
    ---
    """

    def _prepare(self):
        for k, v in groups.items():
            for val in v:
                self.insert1((k,) + val, skip_duplicates=True)

    def load(self, key, preprocessor=None, prefix=''):
        file_key = prefix + 'file_name'
        if preprocessor is None:
            if 'preprocessing' in key:
                preprocessor = preprocessors[key['preprocessing']]
            else:
                raise KeyError('No preprocessor specified')

        with h5py.File(key[file_key]) as fid:
            X = np.asarray(fid['stack'], dtype=float)
            return preprocessor(X.squeeze())


@schema
class VoxelSize(dj.Lookup):
    definition = """
    ->StackGroup
    vx          : int # x voxel size
    vy          : int # y voxel size
    vz          : int # z voxel size
    ---
    """

    contents = [
        ('manolis082015', 17, 17, 15),
        ('twitch102015', 17, 17, 15),
        ('sr101', 17, 17, 15),
    ]


@schema
class CellLocations(dj.Manual):
    definition = """
    -> Stacks
    labeller            : varchar(100) # descriptor of the labelling person/algorithm
    ---
    cells               : longblob     # integer array with cell locations cells x 3
    """


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
                                     itertools.product(zip(range(2, 10, 2), range(2, 10, 2)), zip(range(2, 10, 2))))


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
        return [(k,) for k in preprocessors]


@schema
@gitlog
@hdf5
class TrainedBSTM(dj.Computed):
    definition = """
    # trained BSTM models

    -> Stacks
    -> CellLocations
    -> VoxelSize
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
    train_auc                : double   # area under the curve
    train_auc_weighted       : double   # area under the curve
    """

    def _make_tuples(self, key):
        X = Stacks().load(key)
        voxel = (VoxelSize() & key).fetch1['vx', 'vy', 'vz']
        b = RankDegenerateBernoulliProcess(voxel,
                                           quadratic_channels=key['quadratic_components'],
                                           linear_channels=key['linear_components'],
                                           common_channels=key['common_components']
                                           )
        cells = (CellLocations() & key).fetch1['cells']

        b.fit(X, cells, maxiter=100)
        key.update(b.parameters)
        key['train_cross_entropy'] = b.cross_entropy(X, cells)
        key['train_auc_weighted'] = b.auc(X, cells, average='weighted')
        key['train_auc'] = b.auc(X, cells, average='macro')
        self.insert1(key)

    def key2BSTM(self, key):
        trained = (self & key).fetch1()
        voxel = key['vx'], key['vy'], key['vz']
        b = RankDegenerateBernoulliProcess(voxel, quadratic_channels=key['quadratic_components'],
                                           linear_channels=key['linear_components'],
                                           common_channels=key['common_components'])
        b.set_parameters(**trained)
        return b

@schema
@gitlog
class ValidationBSTM(dj.Computed):
    definition = """
    # trained BSTM models
    -> TrainedBSTM
    val_file_name          : varchar(100)  # filename
    val_labeller           : varchar(100) # descriptor of the labelling person/algorithm
    ---

    val_cross_entropy      : double   # in Bits/component
    val_auc                : double   # area under the curve
    val_auc_weighted       : double   # area under the curve
    """

    @property
    def populated_from(self):
        return TrainedBSTM().project() * TrainedBSTM().project(val_file_name='file_name', val_labeller='labeller') \
               - 'file_name=val_file_name OR labeller=val_labeller'

    def _make_tuples(self, key):
        b = TrainedBSTM().key2BSTM(key)
        X = Stacks().load(key, prefix='val_')
        cells = (CellLocations().project(val_file_name='file_name', val_labeller='labeller', cells='cells')
                 & key).fetch1['cells']

        key['val_cross_entropy'] = b.cross_entropy(X, cells)
        key['val_auc_weighted'] = b.auc(X, cells, average='weighted')
        key['val_auc'] = b.auc(X, cells, average='macro')
        self.insert1(key)


@schema
@gitlog
class TestedBSTM(dj.Computed):
    definition = """
    # trained BSTM models
    -> TrainedBSTM
    -> ValidationBSTM
    test_file_name          : varchar(100)  # filename
    test_labeller           : varchar(100)  # descriptor of the labelling person/algorithm
    ---

    test_cross_entropy      : double   # in Bits/component
    test_auc                : double   # area under the curve
    test_auc_weighted       : double   # area under the curve
    """

    @property
    def populated_from(self):
        best = (Stacks() * CellLocations() * VoxelSize()).aggregate(ValidationBSTM(),
                                                                    max_auc_weighted='MAX(val_auc_weighted)')
        selection = ValidationBSTM() * best & 'max_auc_weighted=val_auc_weighted'

        return selection * TrainedBSTM().project(test_file_name='file_name', test_labeller='labeller') \
            & """((file_name != val_file_name OR labeller != val_labeller) AND \
                  (file_name != test_file_name OR labeller != test_labeller) AND \
                  (val_file_name != test_file_name OR val_labeller != test_labeller))
            """


    def _make_tuples(self, key):
        b = TrainedBSTM().key2BSTM(key)
        X = Stacks().load(key, prefix='test_')
        voxel = (VoxelSize() & key).fetch1['vx', 'vy', 'vz']
        cells = (CellLocations().project(test_file_name='file_name', test_labeller='labeller', cells='cells')
                 & key).fetch1['cells']

        key['test_cross_entropy'] = b.cross_entropy(X, cells)
        key['test_auc_weighted'] = b.auc(X, cells, average='weighted')
        key['test_auc'] = b.auc(X, cells, average='macro')
        self.insert1(key)


@schema
@gitlog
class BSTMCellScoreMap(dj.Computed):
    definition = """
    ->TrainedBSTM
    test_file_name          : varchar(100)  # filename
    test_labeller           : varchar(100) # descriptor of the labelling person/algorithm
    ---
    test_cross_entropy      : double   # in Bits/component
    test_auc                : double   # in Bits/component
    test_auc_weighted       : double   # in Bits/component

    """

    class ProbabilityMapSlice(dj.Part):
        definition = """
        -> BSTMCellScoreMap
        slice_idx            : int # index into the depth dimension of the stack
        ---
        map                  : longblob # actual slice
        """

    @property
    def populated_from(self):
        # get the best models over parameters and restarts in training (sloppy model selection, I know)
        best = (Stacks() * CellLocations() * VoxelSize()).aggregate(TrainedBSTM(), max_aucw='MAX(train_auc_weighted)')
        # get the parameters for those models
        models = best * TrainedBSTM() & 'train_auc_weighted = max_aucw'
        # get all stacks within a group that the algorithm was not trained on
        return models * (Stacks() * CellLocations() * VoxelSize()).project(test_file_name='file_name',
                                                                           test_labeller='labeller') \
               - 'file_name = test_file_name'

    def _make_tuples(self, key):
        b = TrainedBSTM().key2BSTM(key)

        X_test = Stacks().load(key, prefix='test_')
        cells_test = (CellLocations().project(test_file_name='file_name', test_labeller='labeller', cells='cells')
                      & key).fetch1['cells']
        P = b.P(X_test, full=True)
        key['test_cross_entropy'] = b.cross_entropy(X_test, cells_test)
        key['test_auc_weighted'] = b.auc(X_test, cells_test, average='weighted')
        key['test_auc'] = b.auc(X_test, cells_test, average='macro')
        self.insert1(key)
        del key['test_cross_entropy']
        del key['test_auc_weighted']
        del key['test_auc']

        pms = self.ProbabilityMapSlice()
        for i in range(P.shape[2]):
            key['slice_idx'] = i
            key['map'] = P[..., i].squeeze()
            pms.insert1(key)


if __name__ == "__main__":
    TrainedBSTM().populate(reserve_jobs=True)
    ValidationBSTM().populate(reserve_jobs=True)
    BSTMCellScoreMap().populate(reserve_jobs=True)
    TestedBSTM().populate(reserve_jobs=True)
