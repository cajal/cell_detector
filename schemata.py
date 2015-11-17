import datajoint as dj
from stack import Stack
from bernoulli import FullBernoulliProcess
from utils import preprocess
schema = dj.schema('datajoint_aod_cell_detection', locals())


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
        ('data/2015-08-25_12-49-41_2015-08-25_13-02-18.h5', 17, 17, 15)
    ]


@schema
class ComponentNumbers(dj.Lookup):
    definition = """
    # number of linear and quadratic components for parameter selection

    linear_components          : int # number of linear components
    quadratic_components       : int # number of quadratic components
    ---

    """

    @property
    def contents(self):
        yield from zip(range(1, 6), range(1, 6))


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
class TrainedFullBernoulliProcess(dj.Computed):
    definition = """
    # Trained full Bernoulli Processes

    -> TrainingFiles
    -> ComponentNumbers
    -> Repetitions
    ---

    u               : longblob # quadratic filters
    w               : longblob # linear filters
    beta            : longblob # quadratic coefficients
    b               : longblob # offsets
    cross_entropy   : double   # in Bits/component
    """

    def _make_tuples(self, key):
        s = Stack(key['file_name'], preprocessor=lambda x: preprocess(x).mean(axis=-1).squeeze())
        voxel = (TrainingFiles() & key).fetch1['vx', 'vy', 'vz']
        b = FullBernoulliProcess(voxel,
                                 quadratic_channels=key['quadratic_components'],
                                 linear_channels=key['linear_components'])
        b.fit(s.X, s.cells)
        key.update(b.parameters)
        key['cross_entropy'] = b.cross_entropy(s.X, s.cells)
        self.insert1(key)


if __name__ == "__main__":
    TrainedFullBernoulliProcess().populate(reserve_jobs=True)
