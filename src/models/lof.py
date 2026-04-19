import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.neighbors import LocalOutlierFactor


class LOFModel:
    def __init__(self, n_neighbors=20, contamination=0.0017):
        self._contamination = contamination
        self._model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            novelty=True,
            n_jobs=-1,
        )

    def fit(self, X_train, y_train=None):
        print(f"  Fitting on {X_train.shape[0]:,} samples (novelty mode, may take a while)...", flush=True)
        self._model.fit(X_train)
        print("  Done.", flush=True)
        return self

    def scores(self, X):
        return -self._model.score_samples(X)

    def predict(self, X):
        s = self.scores(X)
        threshold = np.percentile(s, 100 * (1 - self._contamination))
        return (s > threshold).astype(int)
