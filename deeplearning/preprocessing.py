# preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
from pathlib import Path
from config import TARGET_COLUMN, DROP_COLS_KEYWORDS


ENCODER_DIR = Path("encoders")
ENCODER_DIR.mkdir(exist_ok=True)

def fit_encoders(df: pd.DataFrame):
    df = df.dropna(subset=[TARGET_COLUMN])

    # Drop IDs and technical columns
    drop_cols = [
        c for c in df.columns
        if any(k in c.lower() for k in DROP_COLS_KEYWORDS)
    ]
    df = df.drop(columns=drop_cols, errors="ignore")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    # Label encoders for categorical columns
    cat_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        cat_encoders[col] = le

    # Target encoder
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(y.astype(str))

    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])

    joblib.dump(cat_encoders, ENCODER_DIR / "cat_encoders.joblib")
    joblib.dump(target_encoder, ENCODER_DIR / "target_encoder.joblib")
    joblib.dump(scaler, ENCODER_DIR / "scaler.joblib")
    joblib.dump(num_cols, ENCODER_DIR / "num_cols.joblib")
    joblib.dump(cat_cols, ENCODER_DIR / "cat_cols.joblib")

    print("✅ Encoders fitted and saved.")
