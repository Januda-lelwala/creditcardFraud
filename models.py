import numpy as np
import warnings
warnings.filterwarnings('ignore')


class IsolationForestModel:
    def __init__(self, n_estimators=100, contamination=0.0017, random_state=42):
        from sklearn.ensemble import IsolationForest
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


class LOFModel:
    def __init__(self, n_neighbors=20, contamination=0.0017):
        from sklearn.neighbors import LocalOutlierFactor
        self._contamination = contamination
        self._model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            novelty=True,
            n_jobs=-1,
        )

    def fit(self, X_train, y_train=None):
        print(f"  Fitting on {X_train.shape[0]:,} samples (novelty mode, this may take a while)...", flush=True)
        self._model.fit(X_train)
        print("  Done.", flush=True)
        return self

    def scores(self, X):
        return -self._model.score_samples(X)

    def predict(self, X):
        s = self.scores(X)
        threshold = np.percentile(s, 100 * (1 - self._contamination))
        return (s > threshold).astype(int)


class OneClassSVMModel:
    def __init__(self, nu=0.0017, gamma=0.1, n_components=300, random_state=42):
        from sklearn.linear_model import SGDOneClassSVM
        from sklearn.kernel_approximation import Nystroem
        from sklearn.pipeline import make_pipeline
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


class KNNModel:
    def __init__(self, n_neighbors=2, metric='euclidean', weights='uniform',
                 p=2, smote_random_state=28):
        from sklearn.neighbors import KNeighborsClassifier
        self._smote_rs = smote_random_state
        self._model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            metric=metric,
            weights=weights,
            p=p,
            n_jobs=-1,
        )

    def fit(self, X_train, y_train):
        from imblearn.over_sampling import SMOTE
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


class XGBoostModel:
    def __init__(self, n_estimators=300, max_depth=17, learning_rate=0.06,
                 subsample=0.8, colsample_bytree=0.8, random_state=42,
                 smote_random_state=28):
        from xgboost import XGBClassifier
        self._smote_rs = smote_random_state
        self._n_estimators = n_estimators
        self._model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            eval_metric='aucpr',
            random_state=random_state,
            n_jobs=-1,
            verbosity=0,
        )

    def fit(self, X_train, y_train):
        from imblearn.over_sampling import SMOTE
        print("  Applying SMOTE...", flush=True)
        X_sm, y_sm = SMOTE(random_state=self._smote_rs).fit_resample(X_train, y_train)
        print(f"  After SMOTE — fraud: {y_sm.sum():,}, normal: {(y_sm == 0).sum():,}", flush=True)
        print(f"  Training XGBoost ({self._n_estimators} rounds)...", flush=True)
        self._model.fit(X_sm, y_sm)
        print("  Done.", flush=True)
        return self

    def scores(self, X):
        return self._model.predict_proba(X)[:, 1]

    def predict(self, X):
        return self._model.predict(X)


class AutoencoderModel:
    def __init__(self, epochs=20, batch_size=256, lr=1e-3,
                 threshold_pct=95, random_state=42):
        self._epochs = epochs
        self._batch_size = batch_size
        self._lr = lr
        self._threshold_pct = threshold_pct
        self._threshold = None
        self._net = None
        self._device = None

        import torch
        torch.manual_seed(random_state)

    def _build_net(self, n_features):
        import torch.nn as nn

        class _Net(nn.Module):
            def __init__(self, n_features):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(n_features, 32), nn.ReLU(),
                    nn.Linear(32, 16),         nn.ReLU(),
                    nn.Linear(16, 8),          nn.ReLU(),
                )
                self.decoder = nn.Sequential(
                    nn.Linear(8, 16),          nn.ReLU(),
                    nn.Linear(16, 32),         nn.ReLU(),
                    nn.Linear(32, n_features),
                )

            def forward(self, x):
                return self.decoder(self.encoder(x))

        return _Net(n_features)

    def fit(self, X_train, y_train):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        self._device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"  Device: {self._device}", flush=True)

        X_normal = X_train[y_train == 0]
        print(f"  Training on {X_normal.shape[0]:,} normal samples for {self._epochs} epochs...", flush=True)

        self._net = self._build_net(X_train.shape[1]).to(self._device)
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self._lr)
        criterion = nn.MSELoss()

        X_t = torch.tensor(X_normal, dtype=torch.float32)
        loader = DataLoader(TensorDataset(X_t, X_t), batch_size=self._batch_size, shuffle=True)

        self._net.train()
        for epoch in range(self._epochs):
            epoch_loss = torch.zeros(1, device=self._device)
            for X_batch, _ in loader:
                X_batch = X_batch.to(self._device)
                optimizer.zero_grad()
                loss = criterion(self._net(X_batch), X_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach() * len(X_batch)
            avg_loss = (epoch_loss / len(X_normal)).item()
            print(f"  Epoch {epoch+1:2d}/{self._epochs}  loss: {avg_loss:.6f}", flush=True)

        self._net.eval()
        train_errors = self._reconstruction_errors(X_normal)
        self._threshold = np.percentile(train_errors, self._threshold_pct)
        print(f"  Reconstruction error threshold ({self._threshold_pct}th pct): {self._threshold:.6f}", flush=True)
        print("  Done.", flush=True)
        return self

    def _reconstruction_errors(self, X_np):
        import torch
        X_t = torch.tensor(X_np, dtype=torch.float32).to(self._device)
        with torch.no_grad():
            X_rec = self._net(X_t).cpu().numpy()
        return np.mean(np.square(X_np - X_rec), axis=1)

    def scores(self, X):
        return self._reconstruction_errors(X)

    def predict(self, X):
        return (self.scores(X) > self._threshold).astype(int)
