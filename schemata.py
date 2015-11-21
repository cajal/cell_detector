import datajoint as dj
from stack import Stack
from bernoulli import FullBernoulliProcess, RankDegenerateBernoulliProcess
from utils import *

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
        ('data/2015-08-25_12-49-41_2015-08-25_13-02-18.h5', 17, 17, 15), # TODO move voxels somewhere else and join training and test data
    ]


@schema
class TestingFiles(dj.Lookup):
    definition = """
    # input testing files

    file_name   : varchar(100)  # filename
    ---
    vx          : int # x voxel size
    vy          : int # y voxel size
    vz          : int # z voxel size
    """

    contents = [
        ('2015-08-25_13-49-54_2015-08-25_13-57-23.h5', 17, 17, 15),
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
        yield from map(tuple, preprocessors.keys())


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

    class GitKey(dj.Part):
        definition = """
        ->TrainedRDBernoulliProcess
        ---
        sha1        : varchar(40)
        branch      : varchar(50)
        modified    : int   # whether there are modified files or not
        """

        def make_tuple(self, key):
            repo = git.Repo('./')
            sha1, branch = repo.head.commit.name_rev.split()
            modified = (repo.git.status().find("modified") > 0) * 1
            key['sha1'] = sha1
            key['branch'] = branch
            key['modified'] = modified
            self.insert1(key)

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
        TrainedRDBernoulliProcess.GitKey().make_tuple(key_sub)

    def plot(self):
        for key in (self & 'quadratic_components = 9' & 'linear_components=9').fetch.as_dict:

            voxel = (TrainingFiles() & key).fetch1['vx', 'vy', 'vz']
            b = RankDegenerateBernoulliProcess(voxel, quadratic_channels=key['quadratic_components'],
                                               linear_channels=key['linear_components'],
                                               common_channels=key['common_components'])
            b.set_parameters(**key)
            print(b)
            s = Stack(key['file_name'], preprocessor=preprocessors[key['preprocessing']])
            b.visualize(s.X, s.cells)


if __name__ == "__main__":
    TrainedRDBernoulliProcess().populate(reserve_jobs=True)
    # TrainedRDBernoulliProcess().plot()
