"""مدل LSTM با PyTorch"""
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from loguru import logger

MODEL_DIR = Path("data/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=3, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out).squeeze(-1)


class LSTMModel:
    name = "lstm"

    def __init__(self, input_size=None, hidden_size=64, num_layers=2,
                 dropout=0.3, seq_len=60, epochs=50, batch_size=64,
                 learning_rate=0.001, patience=10):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.seq_len = seq_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.patience = patience
        self.scaler: StandardScaler = None
        self.net: _LSTMNet = None

    # ---------- internals ----------
    def _make_sequences(self, X: np.ndarray, y: np.ndarray = None):
        n = len(X)
        if n < self.seq_len:
            raise ValueError(f"Not enough samples ({n} < {self.seq_len})")
        Xs = np.stack([X[i:i + self.seq_len] for i in range(n - self.seq_len + 1)])
        ys = None
        if y is not None:
            ys = y[self.seq_len - 1:].astype(np.float32)
        return Xs, ys

    # ---------- training ----------
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.scaler = StandardScaler()
        Xtr = self.scaler.fit_transform(X_train.values if hasattr(X_train, "values") else X_train)
        ytr = np.asarray(y_train).astype(np.float32)

        if self.input_size is None:
            self.input_size = Xtr.shape[1]

        Xseq, yseq = self._make_sequences(Xtr, ytr)
        logger.info(f"[{self.name}] Training on {len(Xseq)} sequences, "
                    f"input_size={self.input_size}, device={DEVICE}")

        # validation
        if X_val is not None and y_val is not None and len(X_val) >= self.seq_len:
            Xv = self.scaler.transform(X_val.values if hasattr(X_val, "values") else X_val)
            yv = np.asarray(y_val).astype(np.float32)
            Xvseq, yvseq = self._make_sequences(Xv, yv)
        else:
            Xvseq, yvseq = None, None

        # tensors
        Xt = torch.tensor(Xseq, dtype=torch.float32, device=DEVICE)
        yt = torch.tensor(yseq, dtype=torch.float32, device=DEVICE)

        # network
        self.net = _LSTMNet(
            self.input_size, self.hidden_size,
            self.num_layers, self.dropout,
        ).to(DEVICE)

        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        criterion = nn.BCEWithLogitsLoss()

        n = len(Xt)
        best_val_loss = float("inf")
        best_state = None
        wait = 0

        for epoch in range(1, self.epochs + 1):
            self.net.train()
            perm = torch.randperm(n, device=DEVICE)
            epoch_loss = 0.0
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                xb = Xt[idx]
                yb = yt[idx]
                optimizer.zero_grad()
                logits = self.net(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(idx)
            epoch_loss /= n

            val_msg = ""
            if Xvseq is not None:
                self.net.eval()
                with torch.no_grad():
                    Xv_t = torch.tensor(Xvseq, dtype=torch.float32, device=DEVICE)
                    yv_t = torch.tensor(yvseq, dtype=torch.float32, device=DEVICE)
                    val_logits = self.net(Xv_t)
                    val_loss = criterion(val_logits, yv_t).item()
                val_msg = f" | val_loss={val_loss:.4f}"

                if val_loss < best_val_loss - 1e-4:
                    best_val_loss = val_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
                    wait = 0
                else:
                    wait += 1

            if epoch % 5 == 0 or epoch == 1:
                logger.info(f"[{self.name}] epoch {epoch:>3}: train_loss={epoch_loss:.4f}{val_msg}")

            if Xvseq is not None and wait >= self.patience:
                logger.info(f"[{self.name}] Early stopping at epoch {epoch}")
                break

        if best_state is not None:
            self.net.load_state_dict(best_state)
            logger.info(f"[{self.name}] Restored best model (val_loss={best_val_loss:.4f})")

        return self

    # ---------- prediction ----------
    def predict_proba(self, X) -> np.ndarray:
        if self.net is None or self.scaler is None:
            raise RuntimeError("Model not fitted")
        Xs = self.scaler.transform(X.values if hasattr(X, "values") else X)
        Xseq, _ = self._make_sequences(Xs, None)

        self.net.eval()
        with torch.no_grad():
            Xt = torch.tensor(Xseq, dtype=torch.float32, device=DEVICE)
            logits = self.net(Xt)
            probs = torch.sigmoid(logits).cpu().numpy()

        # Pad ابتدای آرایه با 0.5 (بدون پیش‌بینی برای اولین seq_len-1 نمونه)
        pad = np.full(self.seq_len - 1, 0.5, dtype=np.float32)
        return np.concatenate([pad, probs])

    def evaluate(self, X, y):
        proba = self.predict_proba(X)
        # حذف نمونه‌های pad شده
        proba_valid = proba[self.seq_len - 1:]
        y_valid = np.asarray(y)[self.seq_len - 1:]
        pred = (proba_valid >= 0.5).astype(int)
        acc = accuracy_score(y_valid, pred)
        try:
            auc = roc_auc_score(y_valid, proba_valid)
        except ValueError:
            auc = float("nan")
        logger.info(f"[{self.name}] acc={acc:.4f} auc={auc:.4f}")
        return {"accuracy": acc, "auc": auc}

    # ---------- persistence ----------
    def save(self, path: str = None):
        path = Path(path) if path else MODEL_DIR / f"{self.name}.pt"
        torch.save({
            "state_dict": self.net.state_dict(),
            "scaler": self.scaler,
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "seq_len": self.seq_len,
        }, path)
        logger.info(f"[{self.name}] Saved -> {path}")

    @classmethod
    def load(cls, path: str = None):
        path = Path(path) if path else MODEL_DIR / f"{cls.name}.pt"
        data = torch.load(path, map_location=DEVICE, weights_only=False)
        instance = cls(
            input_size=data["input_size"],
            hidden_size=data["hidden_size"],
            num_layers=data["num_layers"],
            dropout=data["dropout"],
            seq_len=data["seq_len"],
        )
        instance.scaler = data["scaler"]
        instance.net = _LSTMNet(
            data["input_size"], data["hidden_size"],
            data["num_layers"], data["dropout"],
        ).to(DEVICE)
        instance.net.load_state_dict(data["state_dict"])
        instance.net.eval()
        logger.info(f"[{cls.name}] Loaded from {path}")
        return instance
