"""ULB / Kaggle Credit Card Fraud Dataset Adapter.

Loads the public European credit card fraud benchmark dataset (284,807 transactions,
492 fraud transactions across ~48 hours) and prepares it for temporal ML evaluation.

Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
Reference: Andrea Dal Pozzolo et al., "Calibrating Probability with Undersampling
           for Unbalanced Classification", IEEE SSCI 2015.

Key Design Decisions (Refined Experimental Protocol):
1. Strict 3-way temporal split: TRAIN (70%) -> CALIBRATION (15%) -> LOCKED TEST (15%).
2. Evaluates raw transaction features (Time, Amount, V1-V28). No artificial merchant or
   customer clustering is invented.
3. Principled ML feature group definitions:
   - "pca": V1 through V28 (28 anonymized PCA dimensions)
   - "amount_time": Time, Amount (raw temporal and transaction amount features)
   - "all": All 30 transaction features
4. Zero future-data leakage: splitting is performed strictly by time order.
"""

import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.contracts.contracts import Transaction


# Principled feature groups for ML ablation studies
ML_FEATURE_GROUPS: Dict[str, List[str]] = {
    "pca": [f"V{i}" for i in range(1, 29)],
    "pca_plus_amount": [f"V{i}" for i in range(1, 29)] + ["Amount"],
    "amount_time": ["Time", "Amount"],
}

ALL_FEATURE_COLS = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]


class KaggleCreditCardAdapter:
    """Loads and transforms the ULB / Kaggle Credit Card Fraud dataset for ML evaluation.

    The adapter handles:
    1. Loading the raw CSV from a local path or auto-downloading via kagglehub.
    2. Enforcing a strict 3-way temporal split: TRAIN (70%), CALIBRATION (15%), TEST (15%).
    3. Extracting feature matrices (X, y) for principled feature ablation studies.
    4. Converting raw rows to project Transaction contracts for pipeline compatibility.
    """

    BASE_TIMESTAMP = datetime(2023, 9, 1, 0, 0, 0, tzinfo=timezone.utc)

    def __init__(self, csv_path: Optional[str] = None, seed: int = 42):
        self.seed = seed
        self._rng = np.random.RandomState(seed)

        if csv_path and Path(csv_path).exists():
            self._csv_path = Path(csv_path)
        else:
            self._csv_path = self._find_or_download()

        self._raw_df: Optional[pd.DataFrame] = None

    @staticmethod
    def create_synthetic_benchmark_matrix(
        n_rows: int = 100, seed: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Generate synthetic 30-feature transaction data for offline testing."""
        rng = np.random.RandomState(seed)
        times = np.sort(rng.uniform(0, 172800, size=n_rows))
        amounts = rng.exponential(scale=88.0, size=n_rows)
        pca_cols = [f"V{i}" for i in range(1, 29)]
        pca_vals = rng.randn(n_rows, 28)

        data = {"Time": times, "Amount": amounts}
        for i, col in enumerate(pca_cols):
            data[col] = pca_vals[:, i]

        labels = np.zeros(n_rows, dtype=np.int32)
        # Distribute positive samples evenly across rows
        labels[::12] = 1
        data["Class"] = labels

        df = pd.DataFrame(data)
        X = df[ALL_FEATURE_COLS].values.astype(np.float64)
        y = labels
        return X, y, df

    @staticmethod
    def _find_or_download() -> Path:
        """Find local cached CSV or auto-download via kagglehub."""
        cache_dir = Path.home() / ".cache" / "kagglehub" / "datasets" / "mlg-ulb" / "creditcardfraud"
        if cache_dir.exists():
            for csv_file in cache_dir.rglob("creditcard.csv"):
                return csv_file

        try:
            import kagglehub
            path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
            csv_file = Path(path) / "creditcard.csv"
            if csv_file.exists():
                return csv_file
            for f in Path(path).rglob("*.csv"):
                return f
            raise FileNotFoundError(f"No CSV found in {path}")
        except ImportError:
            raise ImportError(
                "kagglehub not installed. Install via: pip install kagglehub\n"
                "Or provide csv_path directly to KaggleCreditCardAdapter(csv_path=...)"
            )

    def load(self) -> pd.DataFrame:
        """Load and preprocess the raw dataset."""
        if self._raw_df is not None:
            return self._raw_df

        df = pd.read_csv(self._csv_path)
        assert "Class" in df.columns, f"Expected 'Class' column, got {df.columns.tolist()}"
        assert len(df) > 200_000, f"Expected 284K+ rows, got {len(df)}"

        df["timestamp"] = df["Time"].apply(
            lambda t: self.BASE_TIMESTAMP + timedelta(seconds=float(t))
        )

        self._raw_df = df
        return df

    def temporal_three_way_split(
        self, train_ratio: float = 0.70, calib_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split dataset strictly by time into TRAIN (70%), CALIBRATION (15%), and TEST (15%)."""
        df = self.load()
        df_sorted = df.sort_values("Time").reset_index(drop=True)

        t1_quantile = train_ratio
        t2_quantile = train_ratio + calib_ratio

        t1_time = df_sorted["Time"].quantile(t1_quantile)
        t2_time = df_sorted["Time"].quantile(t2_quantile)

        train_df = df_sorted[df_sorted["Time"] <= t1_time].copy()
        calib_df = df_sorted[(df_sorted["Time"] > t1_time) & (df_sorted["Time"] <= t2_time)].copy()
        test_df = df_sorted[df_sorted["Time"] > t2_time].copy()

        return train_df, calib_df, test_df

    def get_feature_matrix(
        self,
        df: Optional[pd.DataFrame] = None,
        feature_subset: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract feature matrix X and labels y for ML training and evaluation."""
        if df is None:
            df = self.load()

        if feature_subset == "pca":
            cols = ML_FEATURE_GROUPS["pca"]
        elif feature_subset == "pca_plus_amount":
            cols = ML_FEATURE_GROUPS["pca_plus_amount"]
        elif feature_subset == "amount_time":
            cols = ML_FEATURE_GROUPS["amount_time"]
        elif feature_subset is None or feature_subset == "all":
            cols = ALL_FEATURE_COLS
        elif isinstance(feature_subset, list):
            cols = feature_subset
        else:
            raise ValueError(f"Unknown feature_subset: {feature_subset}")

        X = df[cols].values.astype(np.float64)
        y = df["Class"].values.astype(np.int32)
        return X, y

    def get_feature_names(self, feature_subset: Optional[str] = None) -> List[str]:
        """Get list of feature column names for given subset."""
        if feature_subset == "pca":
            return list(ML_FEATURE_GROUPS["pca"])
        elif feature_subset == "pca_plus_amount":
            return list(ML_FEATURE_GROUPS["pca_plus_amount"])
        elif feature_subset == "amount_time":
            return list(ML_FEATURE_GROUPS["amount_time"])
        elif feature_subset is None or feature_subset == "all":
            return list(ALL_FEATURE_COLS)
        elif isinstance(feature_subset, list):
            return list(feature_subset)
        else:
            raise ValueError(f"Unknown feature_subset: {feature_subset}")

    def convert_to_transactions(
        self, df: Optional[pd.DataFrame] = None
    ) -> List[Transaction]:
        """Convert DataFrame rows to project Transaction contract objects for system integration."""
        if df is None:
            df = self.load()
        else:
            df = df.copy()

        if "timestamp" not in df.columns:
            df["timestamp"] = df["Time"].apply(
                lambda t: self.BASE_TIMESTAMP + timedelta(seconds=float(t))
            )

        transactions = []
        for idx, row in df.iterrows():
            payload = f"RW:{idx}:{row['Time']}:{row['Amount']}"
            digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

            tx = Transaction(
                transaction_id=f"TX-RW-{digest}",
                timestamp=row["timestamp"],
                merchant_id="RW-MERCHANT-GLOBAL",
                customer_id=f"CUST-RW-{digest[:6]}",
                amount=float(row["Amount"]),
                payment_method="card",
                country="EU",
                device_id=f"DEV-RW-{digest[6:]}",
            )
            transactions.append(tx)

        return transactions

    def compute_dataset_hash(self) -> str:
        """Compute SHA-256 integrity hash of raw dataset file."""
        sha = hashlib.sha256()
        with open(self._csv_path, "rb") as f:
            while chunk := f.read(65536):
                sha.update(chunk)
        return sha.hexdigest()

    def get_dataset_manifest(self) -> Dict[str, Any]:
        """Return dataset manifest metadata for audit provenance."""
        df = self.load()
        train_df, calib_df, test_df = self.temporal_three_way_split()

        return {
            "dataset_name": "ULB / Kaggle Credit Card Fraud Detection Benchmark",
            "reference": "Andrea Dal Pozzolo et al., IEEE SSCI 2015",
            "dataset_sha256": self.compute_dataset_hash(),
            "dataset_filename": "creditcard.csv",
            "dataset_source": "kaggle/mlg-ulb/creditcardfraud",
            "total_transactions": len(df),
            "total_fraud": int(df["Class"].sum()),
            "total_legitimate": int((df["Class"] == 0).sum()),
            "fraud_rate": float(df["Class"].mean()),
            "time_span_hours": float((df["Time"].max() - df["Time"].min()) / 3600),
            "splits": {
                "train": {
                    "count": len(train_df),
                    "fraud_count": int(train_df["Class"].sum()),
                    "ratio": float(len(train_df) / len(df)),
                    "min_time": float(train_df["Time"].min()),
                    "max_time": float(train_df["Time"].max()),
                },
                "calibration": {
                    "count": len(calib_df),
                    "fraud_count": int(calib_df["Class"].sum()),
                    "ratio": float(len(calib_df) / len(df)),
                    "min_time": float(calib_df["Time"].min()),
                    "max_time": float(calib_df["Time"].max()),
                },
                "test": {
                    "count": len(test_df),
                    "fraud_count": int(test_df["Class"].sum()),
                    "ratio": float(len(test_df) / len(df)),
                    "min_time": float(test_df["Time"].min()),
                    "max_time": float(test_df["Time"].max()),
                },
            },
        }
