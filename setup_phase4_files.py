# setup_phase4_files.py
"""ساخت فایل‌های فاز ۴: مدل‌های Ensemble"""
from pathlib import Path

FILES = {}

# ---------- src/models/dataset.py ----------
FILES["src/models/dataset.py"] = '''"""آماده‌سازی داده برای آموزش مدل‌ها"""
from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger

PROCESSED_DIR = Path("data/processed")

# ستون‌های خام که نباید به‌عنوان ویژگی استفاده شوند
RAW_COLS = ["open", "high", "low", "close"]


def load_features(symbol: str, interval: str) -> pd.DataFrame:
    """بارگذاری ویژگی‌های ساخته‌شده"""
    path = PROCESSED_DIR / f"{symbol}_{interval}_features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Features not found: {path}")
    df = pd.read_parquet(path)
    logger.info(f"Loaded {len(df)} rows, {df.shape[1]} cols from {path}")
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """لیست ستون‌های ویژگی (بدون OHLC خام)"""
    cols = [c for c in df.columns if c not in RAW_COLS]
    return cols


def create_target(df: pd.DataFrame, horizon: int = 1) -> pd.Series:
    """هدف: جهت قیمت در کندل بعدی (1=صعودی، 0=نزولی)"""
    future_close = df["close"].shift(-horizon)
    target = (future_close > df["close"]).astype(int)
    target.name = "target"
    return target


def prepare_dataset(symbol: str, interval: str, horizon: int = 1):
    """آماده‌سازی کامل X، y و لیست ویژگی‌ها"""
    df = load_features(symbol, interval)
    y = create_target(df, horizon=horizon)

    # حذف آخرین ردیف(ها) که target آن‌ها NaN است
    valid = y.notna()
    df = df[valid]
    y = y[valid].astype(int)

    feature_cols = get_feature_columns(df)
    X = df[feature_cols]

    logger.info(
        f"Dataset: X={X.shape}, y={y.shape}, "
        f"class balance: up={y.mean()*100:.1f}%, down={(1-y.mean())*100:.1f}%"
    )
    return X, y, feature_cols


def train_val_test_split(X, y, train_ratio=0.70, val_ratio=0.15):
    """تقسیم زمانی (بدون shuffle برای جلوگیری از Look-Ahead)"""
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]
    X_val = X.iloc[train_end:val_end]
    y_val = y.iloc[train_end:val_end]
    X_test = X.iloc[val_end:]
    y_test = y.iloc[val_end:]

    logger.info(
        f"Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
'''

# ---------- src/models/xgboost_model.py ----------
FILES["src/models/xgboost_model.py"] = '''"""مدل XGBoost"""
from pathlib import Path
import joblib
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score
from loguru import logger

MODEL_DIR = Path("data/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


class XGBoostModel:
    name = "xgboost"

    def __init__(self, **params):
        defaults = dict(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
        )
        defaults.update(params)
        self.model = xgb.XGBClassifier(**defaults)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        if X_val is not None and y_val is not None:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
        else:
            self.model.fit(X_train, y_train)
        logger.info(f"[{self.name}] Trained")
        return self

    def predict_proba(self, X) -> np.ndarray:
        """احتمال کلاس 1 (صعودی)"""
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X, y):
        proba = self.predict_proba(X)
        pred = (proba >= 0.5).astype(int)
        acc = accuracy_score(y, pred)
        try:
            auc = roc_auc_score(y, proba)
        except ValueError:
            auc = float("nan")
        logger.info(f"[{self.name}] acc={acc:.4f} auc={auc:.4f}")
        return {"accuracy": acc, "auc": auc}

    def save(self, path: str = None):
        path = Path(path) if path else MODEL_DIR / f"{self.name}.pkl"
        joblib.dump(self.model, path)
        logger.info(f"[{self.name}] Saved -> {path}")

    @classmethod
    def load(cls, path: str = None):
        path = Path(path) if path else MODEL_DIR / f"{cls.name}.pkl"
        instance = cls()
        instance.model = joblib.load(path)
        logger.info(f"[{cls.name}] Loaded from {path}")
        return instance
'''

# ---------- src/models/lightgbm_model.py ----------
FILES["src/models/lightgbm_model.py"] = '''"""مدل LightGBM"""
from pathlib import Path
import joblib
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score
from loguru import logger

MODEL_DIR = Path("data/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


class LightGBMModel:
    name = "lightgbm"

    def __init__(self, **params):
        defaults = dict(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            num_leaves=31,
            objective="binary",
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        defaults.update(params)
        self.model = lgb.LGBMClassifier(**defaults)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        if X_val is not None and y_val is not None:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
            )
        else:
            self.model.fit(X_train, y_train)
        logger.info(f"[{self.name}] Trained")
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X, y):
        proba = self.predict_proba(X)
        pred = (proba >= 0.5).astype(int)
        acc = accuracy_score(y, pred)
        try:
            auc = roc_auc_score(y, proba)
        except ValueError:
            auc = float("nan")
        logger.info(f"[{self.name}] acc={acc:.4f} auc={auc:.4f}")
        return {"accuracy": acc, "auc": auc}

    def save(self, path: str = None):
        path = Path(path) if path else MODEL_DIR / f"{self.name}.pkl"
        joblib.dump(self.model, path)
        logger.info(f"[{self.name}] Saved -> {path}")

    @classmethod
    def load(cls, path: str = None):
        path = Path(path) if path else MODEL_DIR / f"{cls.name}.pkl"
        instance = cls()
        instance.model = joblib.load(path)
        logger.info(f"[{cls.name}] Loaded from {path}")
        return instance
'''

# ---------- src/models/random_forest_model.py ----------
FILES["src/models/random_forest_model.py"] = '''"""مدل Random Forest"""
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from loguru import logger

MODEL_DIR = Path("data/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


class RandomForestModel:
    name = "random_forest"

    def __init__(self, **params):
        defaults = dict(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        defaults.update(params)
        self.model = RandomForestClassifier(**defaults)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.model.fit(X_train, y_train)
        logger.info(f"[{self.name}] Trained")
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X, y):
        proba = self.predict_proba(X)
        pred = (proba >= 0.5).astype(int)
        acc = accuracy_score(y, pred)
        try:
            auc = roc_auc_score(y, proba)
        except ValueError:
            auc = float("nan")
        logger.info(f"[{self.name}] acc={acc:.4f} auc={auc:.4f}")
        return {"accuracy": acc, "auc": auc}

    def save(self, path: str = None):
        path = Path(path) if path else MODEL_DIR / f"{self.name}.pkl"
        joblib.dump(self.model, path)
        logger.info(f"[{self.name}] Saved -> {path}")

    @classmethod
    def load(cls, path: str = None):
        path = Path(path) if path else MODEL_DIR / f"{cls.name}.pkl"
        instance = cls()
        instance.model = joblib.load(path)
        logger.info(f"[{cls.name}] Loaded from {path}")
        return instance
'''

# ---------- src/models/lstm_model.py ----------
FILES["src/models/lstm_model.py"] = '''"""مدل LSTM با PyTorch"""
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
'''

# ---------- src/models/ensemble.py ----------
FILES["src/models/ensemble.py"] = '''"""ترکیب مدل‌ها با وزن‌دهی ساده"""
from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger

MODEL_DIR = Path("data/models")

DEFAULT_WEIGHTS = {
    "xgboost": 0.35,
    "lightgbm": 0.35,
    "random_forest": 0.15,
    "lstm": 0.15,
}


class Ensemble:
    def __init__(self, models: dict, weights: dict = None):
        """
        models: دیکشنری {name: model_instance}
        weights: دیکشنری وزن‌ها (اگر ندهید، از DEFAULT_WEIGHTS استفاده می‌شود)
        """
        self.models = models
        weights = weights or DEFAULT_WEIGHTS
        # نرمال‌سازی وزن‌ها
        total = sum(weights.get(n, 0) for n in models)
        self.weights = {n: weights.get(n, 0) / total for n in models}
        logger.info(f"Ensemble weights: {self.weights}")

    def predict_proba(self, X) -> np.ndarray:
        """پیش‌بینی ترکیبی"""
        result = np.zeros(len(X), dtype=np.float32)
        for name, model in self.models.items():
            p = model.predict_proba(X)
            w = self.weights[name]
            result += w * p
            logger.debug(f"  {name}: w={w:.3f}, mean={p.mean():.4f}")
        return result

    @classmethod
    def load_all(cls, weights: dict = None, include_lstm: bool = True):
        """بارگذاری همه مدل‌ها از دیسک"""
        from src.models.xgboost_model import XGBoostModel
        from src.models.lightgbm_model import LightGBMModel
        from src.models.random_forest_model import RandomForestModel
        from src.models.lstm_model import LSTMModel

        models = {}
        try:
            models["xgboost"] = XGBoostModel.load()
        except Exception as e:
            logger.warning(f"XGBoost load failed: {e}")
        try:
            models["lightgbm"] = LightGBMModel.load()
        except Exception as e:
            logger.warning(f"LightGBM load failed: {e}")
        try:
            models["random_forest"] = RandomForestModel.load()
        except Exception as e:
            logger.warning(f"RandomForest load failed: {e}")
        if include_lstm:
            try:
                models["lstm"] = LSTMModel.load()
            except Exception as e:
                logger.warning(f"LSTM load failed: {e}")

        if not models:
            raise RuntimeError("No models could be loaded")

        return cls(models, weights=weights)
'''

# ---------- src/models/__init__.py ----------
FILES["src/models/__init__.py"] = '''from src.models.dataset import (
    prepare_dataset,
    train_val_test_split,
    get_feature_columns,
    load_features,
)
from src.models.xgboost_model import XGBoostModel
from src.models.lightgbm_model import LightGBMModel
from src.models.random_forest_model import RandomForestModel
from src.models.lstm_model import LSTMModel
from src.models.ensemble import Ensemble

__all__ = [
    "prepare_dataset",
    "train_val_test_split",
    "get_feature_columns",
    "load_features",
    "XGBoostModel",
    "LightGBMModel",
    "RandomForestModel",
    "LSTMModel",
    "Ensemble",
]
'''

# ---------- scripts/train_models.py ----------
FILES["scripts/train_models.py"] = '''"""آموزش همه مدل‌ها"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from loguru import logger

from src.models import (
    prepare_dataset,
    train_val_test_split,
    XGBoostModel,
    LightGBMModel,
    RandomForestModel,
    LSTMModel,
    Ensemble,
)
from src.utils.config_loader import load_config
from src.utils.logger import setup_logger


def main():
    setup_logger("train")
    config = load_config()
    mc = config["models"]

    symbol = "BTCUSDT"
    interval = "1h"
    logger.info(f"=== Training for {symbol} {interval} ===")

    # داده
    X, y, feature_cols = prepare_dataset(symbol, interval, horizon=1)
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = train_val_test_split(X, y)

    results = {}

    # ---------- XGBoost ----------
    logger.info("--- XGBoost ---")
    xgb_model = XGBoostModel(**mc["xgboost"])
    xgb_model.fit(X_tr, y_tr, X_va, y_va)
    xgb_model.evaluate(X_va, y_va)
    results["xgboost"] = xgb_model.evaluate(X_te, y_te)
    xgb_model.save()

    # ---------- LightGBM ----------
    logger.info("--- LightGBM ---")
    lgb_model = LightGBMModel(**mc["lightgbm"])
    lgb_model.fit(X_tr, y_tr, X_va, y_va)
    lgb_model.evaluate(X_va, y_va)
    results["lightgbm"] = lgb_model.evaluate(X_te, y_te)
    lgb_model.save()

    # ---------- Random Forest ----------
    logger.info("--- Random Forest ---")
    rf_model = RandomForestModel(**mc["random_forest"])
    rf_model.fit(X_tr, y_tr)
    rf_model.evaluate(X_va, y_va)
    results["random_forest"] = rf_model.evaluate(X_te, y_te)
    rf_model.save()

    # ---------- LSTM ----------
    logger.info("--- LSTM ---")
    lstm_cfg = mc["lstm"]
    lstm_model = LSTMModel(
        input_size=X_tr.shape[1],
        hidden_size=lstm_cfg["hidden_size"],
        num_layers=lstm_cfg["num_layers"],
        dropout=lstm_cfg["dropout"],
        seq_len=lstm_cfg["sequence_length"],
        epochs=lstm_cfg["epochs"],
        batch_size=lstm_cfg["batch_size"],
        learning_rate=lstm_cfg["learning_rate"],
    )
    lstm_model.fit(X_tr, y_tr, X_va, y_va)
    lstm_model.evaluate(X_va, y_va)
    results["lstm"] = lstm_model.evaluate(X_te, y_te)
    lstm_model.save()

    # ---------- Ensemble ----------
    logger.info("--- Ensemble (Test set) ---")
    ensemble = Ensemble({
        "xgboost": xgb_model,
        "lightgbm": lgb_model,
        "random_forest": rf_model,
        "lstm": lstm_model,
    })
    proba = ensemble.predict_proba(X_te)

    # برای LSTM، seq_len-1 نمونه اول اعتبار ندارند
    seq_len = lstm_cfg["sequence_length"]
    valid_start = seq_len - 1
    proba_valid = proba[valid_start:]
    y_valid = y_te.values[valid_start:]

    from sklearn.metrics import accuracy_score, roc_auc_score
    pred = (proba_valid >= 0.5).astype(int)
    ens_acc = accuracy_score(y_valid, pred)
    try:
        ens_auc = roc_auc_score(y_valid, proba_valid)
    except ValueError:
        ens_auc = float("nan")
    results["ensemble"] = {"accuracy": ens_acc, "auc": ens_auc}
    logger.info(f"[ensemble] acc={ens_acc:.4f} auc={ens_auc:.4f}")

    # ---------- خلاصه ----------
    logger.info("")
    logger.info("=" * 60)
    logger.info("Test Set Results (summary)")
    logger.info("=" * 60)
    for name, m in results.items():
        logger.info(f"  {name:<15}: acc={m['accuracy']:.4f}  auc={m['auc']:.4f}")
    logger.info("=" * 60)
    logger.info("")
    logger.info("All models trained and saved successfully!")


if __name__ == "__main__":
    main()
'''

# ---------- tests/test_models.py ----------
FILES["tests/test_models.py"] = '''"""تست مدل‌های آموزش‌دیده"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from loguru import logger

from src.models import (
    prepare_dataset,
    train_val_test_split,
    XGBoostModel,
    LightGBMModel,
    RandomForestModel,
    LSTMModel,
    Ensemble,
)
from src.utils.logger import setup_logger

setup_logger("test")


def main():
    print()
    print("=" * 60)
    print("Testing All Models")
    print("=" * 60)

    X, y, _ = prepare_dataset("BTCUSDT", "1h")
    (_, _), (_, _), (X_te, y_te) = train_val_test_split(X, y)

    # بارگذاری
    xgb_model = XGBoostModel.load()
    lgb_model = LightGBMModel.load()
    rf_model = RandomForestModel.load()
    lstm_model = LSTMModel.load()

    print("[OK] All models loaded")

    # پیش‌بینی
    print()
    print("Individual models (test set):")
    for name, m in [("xgboost", xgb_model), ("lightgbm", lgb_model),
                    ("random_forest", rf_model), ("lstm", lstm_model)]:
        proba = m.predict_proba(X_te)
        if name == "lstm":
            seq = lstm_model.seq_len
            pv = proba[seq - 1:]
            yv = y_te.values[seq - 1:]
        else:
            pv = proba
            yv = y_te.values
        pred = (pv >= 0.5).astype(int)
        acc = accuracy_score(yv, pred)
        try:
            auc = roc_auc_score(yv, pv)
        except ValueError:
            auc = float("nan")
        print(f"  {name:<15}: acc={acc:.4f} auc={auc:.4f}")

    # Ensemble
    print()
    print("Ensemble (test set):")
    ens = Ensemble({
        "xgboost": xgb_model,
        "lightgbm": lgb_model,
        "random_forest": rf_model,
        "lstm": lstm_model,
    })
    proba = ens.predict_proba(X_te)
    seq = lstm_model.seq_len
    pv = proba[seq - 1:]
    yv = y_te.values[seq - 1:]
    pred = (pv >= 0.5).astype(int)
    acc = accuracy_score(yv, pred)
    try:
        auc = roc_auc_score(yv, pv)
    except ValueError:
        auc = float("nan")
    print(f"  {'ensemble':<15}: acc={acc:.4f} auc={auc:.4f}")

    print()
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
'''


def main():
    root = Path(".")
    created = 0
    for file_path, content in FILES.items():
        path = root / file_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        created += 1
        print(f"  + {file_path}")
    print()
    print("=" * 50)
    print(f"Created {created} files successfully!")
    print("=" * 50)
    print()
    print("Next steps:")
    print("  1. python scripts/train_models.py")
    print("  2. python tests/test_models.py")
    print()


if __name__ == "__main__":
    main()