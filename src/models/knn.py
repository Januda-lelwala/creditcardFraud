import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE


class KNNModel:
    def __init__(self, n_neighbors=2, metric='euclidean', weights='uniform',
                 p=2, smote_random_state=28):
        self._smote_rs = smote_random_state
        self._model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            metric=metric,
            weights=weights,
            p=p,
            n_jobs=-1,
        )

    def fit(self, X_train, y_train):
        print("  Applying SMOTE...", flush=True)
        X_sm, y_sm = SMOTE(random_state=self._smote_rs).fit_resample(X_train, y_train)
        print(f"  After SMOTE — fraud: {y_sm.sum():,}, normal: {(y_sm == 0).sum():,}", flush=True)
        print(f"  Fitting KNN on {X_sm.shape[0]:,} samples...", flush=True)
        self._model.fit(X_sm, y_sm)
        print("  Done.", flush=True)
        return self

    def scores(self, X):
        return self._model.predict_proba(X)[:, 1]

    def predict(self, X):
        return self._model.predict(X)
