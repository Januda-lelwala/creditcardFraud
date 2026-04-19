import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import IsolationForest


class IsolationForestModel:
    def __init__(self, n_estimators=100, contamination=0.0017, random_state=42):
        self._model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(self, X_train, y_train=None):
        print(f"  Fitting on {X_train.shape[0]:,} samples...", flush=True)
        self._model.fit(X_train)
        print("  Done.", flush=True)
        return self

    def scores(self, X):
        return -self._model.score_samples(X)

    def predict(self, X):
        return (self._model.predict(X) == -1).astype(int)
