"""تشخیص Drift در توزیع داده‌ها"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from loguru import logger

DRIFT_DIR = Path("data/drift")
DRIFT_DIR.mkdir(parents=True, exist_ok=True)

REFERENCE_FILE = DRIFT_DIR / "reference_stats.json"


def compute_reference(df: pd.DataFrame, feature_cols: list, name: str = "reference"):
    ref = {}
    for col in feature_cols:
        if col in df.columns:
            vals = df[col].dropna().values
            if len(vals) == 0:
                continue
            ref[col] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "quantiles": [float(q) for q in np.quantile(vals, [0.1, 0.25, 0.5, 0.75, 0.9])],
                "sample": vals[-500:].tolist() if len(vals) >= 500 else vals.tolist(),
            }
    path = DRIFT_DIR / f"{name}_stats.json"
    path.write_text(json.dumps(ref), encoding="utf-8")
    logger.info(f"Saved reference stats: {path} ({len(ref)} features)")


def detect_drift(df: pd.DataFrame, feature_cols: list,
                 reference_name: str = "reference",
                 threshold: float = 0.01) -> dict:
    ref_path = DRIFT_DIR / f"{reference_name}_stats.json"
    if not ref_path.exists():
        return {"has_drift": False, "reason": "no reference"}

    ref = json.loads(ref_path.read_text(encoding="utf-8"))
    drifts = []
    details = {}

    for col in feature_cols:
        if col not in ref or col not in df.columns:
            continue
        ref_sample = np.array(ref[col]["sample"])
        curr_sample = df[col].dropna().values[-500:]
        if len(curr_sample) < 50 or len(ref_sample) < 50:
            continue
        try:
            stat, p = ks_2samp(ref_sample, curr_sample)
            details[col] = {"ks_stat": float(stat), "p_value": float(p)}
            if p < threshold:
                drifts.append(col)
        except Exception:
            pass

    has_drift = len(drifts) > 0
    return {
        "has_drift": has_drift,
        "drifted_features": drifts,
        "drift_ratio": len(drifts) / max(len(feature_cols), 1),
        "details": details,
    }


def check_performance_drift(current_accuracy: float,
                            reference_accuracy: float,
                            threshold: float = 0.05) -> dict:
    drop = reference_accuracy - current_accuracy
    return {
        "has_drift": drop > threshold,
        "current_accuracy": current_accuracy,
        "reference_accuracy": reference_accuracy,
        "drop": drop,
        "threshold": threshold,
    }
