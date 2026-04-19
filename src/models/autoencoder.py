import numpy as np
import warnings
warnings.filterwarnings('ignore')


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
        print(f"  Threshold ({self._threshold_pct}th pct of normal errors): {self._threshold:.6f}", flush=True)
        print("  Done.", flush=True)
        return self

    def save(self, path='models/autoencoder.pth'):
        import os, torch
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({'state_dict': self._net.state_dict(), 'threshold': self._threshold}, path)
        print(f"  Model saved → {path}", flush=True)

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
