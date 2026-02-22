"""
data/preprocessing.py
=====================
Feature engineering, normalisation, and sequence construction
for CICIoT2023, Edge-IIoTset, and N-BaIoT datasets.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

logger = logging.getLogger(__name__)

# ── Dataset-specific configurations ──────────────────────────────────────

DATASET_CONFIGS: Dict[str, dict] = {
    "ciciot2023": {
        "label_col": "label",
        "drop_cols": ["flow_id", "src_ip", "dst_ip", "timestamp"],
        "n_top_features": 47,
        "binary_benign": "BenignTraffic",
    },
    "edge_iiotset": {
        "label_col": "Attack_type",
        "drop_cols": ["frame.time", "ip.src_host", "ip.dst_host"],
        "n_top_features": 61,
        "binary_benign": "Normal",
    },
    "nbaiot": {
        "label_col": "label",
        "drop_cols": [],
        "n_top_features": 115,
        "binary_benign": "benign",
    },
}


class IoTPreprocessor:
    """
    Preprocesses raw IoT dataset CSV files into normalised numpy arrays
    suitable for federated training.

    Parameters
    ----------
    dataset : str
        One of ``ciciot2023``, ``edge_iiotset``, ``nbaiot``.
    n_top_features : int or None
        If set, select this many features by mutual information score.
        If None, use all numeric features.
    sequence_length : int
        Number of consecutive flow records per sequence window.
    """

    def __init__(
        self,
        dataset: str,
        n_top_features: int | None = None,
        sequence_length: int = 32,
    ) -> None:
        if dataset not in DATASET_CONFIGS:
            raise ValueError(
                f"Unknown dataset '{dataset}'. "
                f"Choose from {list(DATASET_CONFIGS.keys())}."
            )
        self.dataset = dataset
        self.cfg = DATASET_CONFIGS[dataset]
        self.n_top_features = n_top_features
        self.sequence_length = sequence_length

        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.selected_features: List[str] = []
        self._fitted = False

    # ── Public API ────────────────────────────────────────────────────────

    def fit_transform(
        self, raw_dir: str | Path
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load, clean, select features, normalise, and split into
        train/val/test arrays.

        Returns
        -------
        X_train, X_val, X_test : np.ndarray, shape (N, seq_len, n_features)
        y_train, y_val, y_test : np.ndarray, shape (N,)
        """
        df = self._load_csvs(raw_dir)
        df = self._clean(df)
        X_raw, y_raw = self._extract_xy(df)

        self.selected_features = self._select_features(X_raw, y_raw)
        X_sel = X_raw[self.selected_features].values

        X_norm = self.scaler.fit_transform(X_sel)
        y_enc  = self.label_encoder.fit_transform(y_raw)
        self._fitted = True

        X_seq, y_seq = self._build_sequences(X_norm, y_enc)
        return self._train_val_test_split(X_seq, y_seq)

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Apply fitted preprocessing to new data (inference / test)."""
        if not self._fitted:
            raise RuntimeError("Call fit_transform() before transform().")
        df = self._clean(df)
        X_raw, y_raw = self._extract_xy(df)
        X_sel  = X_raw[self.selected_features].values
        X_norm = self.scaler.transform(X_sel)
        y_enc  = self.label_encoder.transform(y_raw)
        return self._build_sequences(X_norm, y_enc)

    def save(self, out_dir: str | Path) -> None:
        """Persist scaler, label encoder, and selected features."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler,        out_dir / "scaler.pkl")
        joblib.dump(self.label_encoder, out_dir / "label_encoder.pkl")
        joblib.dump(self.selected_features, out_dir / "features.pkl")
        logger.info("Preprocessor saved to %s", out_dir)

    @classmethod
    def load(cls, dataset: str, out_dir: str | Path) -> "IoTPreprocessor":
        """Restore a fitted preprocessor from disk."""
        out_dir = Path(out_dir)
        obj = cls(dataset)
        obj.scaler            = joblib.load(out_dir / "scaler.pkl")
        obj.label_encoder     = joblib.load(out_dir / "label_encoder.pkl")
        obj.selected_features = joblib.load(out_dir / "features.pkl")
        obj._fitted = True
        return obj

    # ── Private helpers ───────────────────────────────────────────────────

    def _load_csvs(self, raw_dir: str | Path) -> pd.DataFrame:
        raw_dir = Path(raw_dir)
        csv_files = list(raw_dir.glob("**/*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {raw_dir}")
        logger.info("Loading %d CSV file(s) from %s", len(csv_files), raw_dir)
        dfs = [pd.read_csv(f, low_memory=False) for f in csv_files]
        return pd.concat(dfs, ignore_index=True)

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        drop = [c for c in self.cfg["drop_cols"] if c in df.columns]
        df   = df.drop(columns=drop)
        df   = df.replace([np.inf, -np.inf], np.nan)
        df   = df.dropna()
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        logger.info("After cleaning: %d rows, %d columns", *df.shape)
        return df

    def _extract_xy(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        label_col = self.cfg["label_col"].lower().replace(" ", "_")
        y = df[label_col].astype(str).str.strip()
        X = df.drop(columns=[label_col]).select_dtypes(include=[np.number])
        return X, y

    def _select_features(
        self, X: pd.DataFrame, y: pd.Series
    ) -> List[str]:
        n = self.n_top_features or X.shape[1]
        n = min(n, X.shape[1])
        if n == X.shape[1]:
            logger.info("Using all %d features", n)
            return list(X.columns)

        logger.info(
            "Selecting top %d features by mutual information...", n
        )
        y_enc = LabelEncoder().fit_transform(y)
        scores = mutual_info_classif(
            X.values, y_enc, random_state=42, n_neighbors=5
        )
        idx = np.argsort(scores)[::-1][:n]
        selected = [X.columns[i] for i in idx]
        logger.info("Selected features: %s", selected[:5])
        return selected

    def _build_sequences(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Slide a window of length ``sequence_length`` over ordered rows.
        Label of sequence = label of last row in window.
        """
        T = self.sequence_length
        N = len(X) - T + 1
        if N <= 0:
            raise ValueError(
                f"Not enough samples ({len(X)}) "
                f"for sequence length {T}."
            )
        X_seq = np.lib.stride_tricks.sliding_window_view(X, (T, X.shape[1]))
        X_seq = X_seq[:, 0, :, :]        # shape: (N, T, n_features)
        y_seq = y[T - 1:]                 # label = last timestep
        return X_seq.astype(np.float32), y_seq.astype(np.int64)

    def _train_val_test_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_ratio: float = 0.2,
        val_ratio:  float = 0.1,
        seed: int = 42,
    ) -> Tuple[
        np.ndarray, np.ndarray,
        np.ndarray, np.ndarray,
        np.ndarray, np.ndarray,
    ]:
        rng  = np.random.default_rng(seed)
        idx  = rng.permutation(len(X))
        n_test = int(len(X) * test_ratio)
        n_val  = int(len(X) * val_ratio)
        test_idx  = idx[:n_test]
        val_idx   = idx[n_test: n_test + n_val]
        train_idx = idx[n_test + n_val:]
        return (
            X[train_idx], y[train_idx],
            X[val_idx],   y[val_idx],
            X[test_idx],  y[test_idx],
        )


# ── CLI entry point ───────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess an IoT IDS dataset for FedCTX-IoT."
    )
    parser.add_argument(
        "--dataset", required=True,
        choices=list(DATASET_CONFIGS.keys()),
        help="Dataset name."
    )
    parser.add_argument("--raw_dir",  required=True,
                        help="Directory containing raw CSV files.")
    parser.add_argument("--out_dir",  required=True,
                        help="Output directory for processed arrays.")
    parser.add_argument("--n_features", type=int, default=None,
                        help="Number of top features to select (default: all).")
    parser.add_argument("--seq_len",    type=int, default=32,
                        help="Sequence window length.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    preprocessor = IoTPreprocessor(
        dataset=args.dataset,
        n_top_features=args.n_features,
        sequence_length=args.seq_len,
    )
    X_tr, y_tr, X_val, y_val, X_te, y_te = preprocessor.fit_transform(
        args.raw_dir
    )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "X_train.npy", X_tr)
    np.save(out / "y_train.npy", y_tr)
    np.save(out / "X_val.npy",   X_val)
    np.save(out / "y_val.npy",   y_val)
    np.save(out / "X_test.npy",  X_te)
    np.save(out / "y_test.npy",  y_te)
    preprocessor.save(out)

    logger.info(
        "Done. Train=%d  Val=%d  Test=%d  Features=%d",
        len(X_tr), len(X_val), len(X_te),
        X_tr.shape[-1],
    )


if __name__ == "__main__":
    main()
