import datajoint as dj
from stack import Stack
from bernoulli import FullBernoulliProcess
from utils import preprocess
schema = dj.schema('datajoint_aod_cell_detection', locals())
import git
import itertools


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
    common_components          : int
    ---

    """

    @property
    def contents(self):
        yield from zip(range(1, 10), range(1, 10), itertools.repeat(10))


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
class TraineRDBernoulliProcess(dj.Computed):
    definition = """
    # Trained ranl degenerate Bernoulli Processes

    -> TrainingFiles
    -> ComponentNumbers
    -> Repetitions
    ---

    u_xy               : longblob # quadratic filters xy
    u_z                : longblob # quadratic filters z
    w_xy               : longblob # linear filters xy
    w_z                : longblob # linear filters z
    beta               : longblob # quadratic coefficients
    gamma              : longblob # linear  coefficients
    b                  : longblob # offsets
    cross_entropy      : double   # in Bits/component
    """

    class GitKey(dj.Part):
        definition = """
        ->TrainedFullBernoulliProcess
        ---
        sha1        : varchar(40)
        branch      : varchar(50)
        modified    : int   # whether there are modified files or not
        """

        def make_tuple(self, key):
            repo = git.Repo('./')
            sha1, branch = repo.head.commit.name_rev.split()
            modified = (repo.git.status().find("modified") > 0)*1
            key['sha1'] = sha1
            key['branch'] = branch
            key['modified'] = modified
            self.insert1(key)


    def _make_tuples(self, key):
        s = Stack(key['file_name'], preprocessor=lambda x: preprocess(x).mean(axis=-1).squeeze())
        voxel = (TrainingFiles() & key).fetch1['vx', 'vy', 'vz']
        b = FullBernoulliProcess(voxel,
                                 quadratic_channels=key['quadratic_components'],
                                 linear_channels=key['linear_components'],
                                 common_channels=key['common_components']
                                 )
        b.fit(s.X, s.cells)
        key.update(b.parameters)
        key['cross_entropy'] = b.cross_entropy(s.X, s.cells)
        self.insert1(key)


if __name__ == "__main__":
    TrainedFullBernoulliProcess().populate(reserve_jobs=True)
