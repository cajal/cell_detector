from multiprocessing.pool import Pool
from pprint import pprint
import numpy as np
from scipy.spatial.distance import pdist
from utils import compute_crange


def split(n, k):
    step = int(np.ceil(n/k))
    cur = 0
    while cur < n:
        yield slice(cur, min(cur + step, n))
        cur += step

class Detector:
    def __init__(self, voxel):
        self._training_data = None
        self.voxel = voxel

    @property
    def training_data(self):
        return self._training_data

    def add_training_data(self, X_train, y_train):
        self._training_data = (np.vstack((self._training_data[0], X_train)),
                               np.hstack((self._training_data[1], y_train)))

    def fit(self, X_train, y_train):
        self._training_data = X_train, y_train

        gamma = 1 / np.median(pdist(X_train, 'euclidean'))
        C = compute_crange(rbf_kernel(X_train, gamma=gamma), basefactors=2 ** np.arange(-3, 6.))

        # Set the parameters by cross-validation
        parameter_grid = [{'kernel': ['rbf'], 'gamma': list(2 ** np.arange(-3, 6.) * gamma), 'C': list(C)}]

        pprint("Considering the following parameters")
        pprint(parameter_grid)

        clf = GridSearchCV(SVC(C=1, probability=True), parameter_grid, cv=5, scoring='accuracy', n_jobs=10,
                           verbose=True)
        clf.fit(X_train, y_train)

        pprint("Found best estimator")
        pprint(clf.best_estimator_)
        pprint(clf.best_score_)

        self.classifier = clf.best_estimator_

    def retrain(self):
        self.fit(*self.training_data)

    def positive_prob(self, X, n_jobs=1):
        if n_jobs == 1:
            prob_all = self.classifier.predict_proba(X)
        else:
            pool = Pool(processes=n_jobs) # depends on available cores
            result = pool.map(self.classifier.predict_proba, [X[idx] for idx in split(len(X), n_jobs)]) # for i in sequence: result[i] = f(i)
            prob_all = np.vstack(result)
            pool.close() # not optimal! but easy
            pool.join()
        return prob_all[:, 1]

