import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import SGDOneClassSVM
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline


class OneClassSVMModel:
    def __init__(self, nu=0.0017, gamma=0.1, n_components=300, random_state=42):
        self._nu = nu
        self._model = make_pipeline(
            Nystroem(kernel='rbf', gamma=gamma, n_components=n_components, random_state=random_state),
            SGDOneClassSVM(nu=nu, random_state=random_state),
        )

    def fit(self, X_train, y_train=None):
        print(f"  Fitting Nystroem + SGD One-Class SVM on {X_train.shape[0]:,} samples...", flush=True)
        self._model.fit(X_train)
        print("  Done.", flush=True)
        return self

    def scores(self, X):
        return -self._model.decision_function(X)

    def predict(self, X):
        s = self.scores(X)
        threshold = np.percentile(s, 100 * (1 - self._nu))
        return (s > threshold).astype(int)
