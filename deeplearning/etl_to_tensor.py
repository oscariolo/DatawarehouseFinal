# etl_to_tensor.py
import pandas as pd
import numpy as np
import torch
import joblib
from pathlib import Path
from config import CACHE_FILE, TARGET_COLUMN, TENSOR_CACHE_FILE

def main():
    print("Starting ETL → Tensor cache")

    BASE_DIR = Path(__file__).resolve().parent
    ENCODER_DIR = BASE_DIR / "encoders"

    # ----------------------------
    # LOAD METADATA
    # ----------------------------
    num_cols = joblib.load(ENCODER_DIR / "num_cols.joblib")
    cat_cols = joblib.load(ENCODER_DIR / "cat_cols.joblib")
    cat_encoders = joblib.load(ENCODER_DIR / "cat_encoders.joblib")
    target_encoder = joblib.load(ENCODER_DIR / "target_encoder.joblib")

    needed_cols = num_cols + cat_cols + [TARGET_COLUMN]

    # ----------------------------
    # LOAD PARQUET (ONCE)
    # ----------------------------
    print("Loading parquet...")
    df = pd.read_parquet(CACHE_FILE, columns=needed_cols)

    # drop rows without target
    df = df.dropna(subset=[TARGET_COLUMN])
    df.reset_index(drop=True, inplace=True)

    print(f"Rows after cleaning: {len(df)}")

    # ----------------------------
    # NUMERIC FEATURES
    # ----------------------------
    print("Encoding numeric features...")
    X_num = df[num_cols].astype(np.float32).values

    # ----------------------------
    # CATEGORICAL FEATURES
    # ----------------------------
    print("Encoding categorical features...")
    X_cat = np.zeros((len(df), len(cat_cols)), dtype=np.int64)

    for i, col in enumerate(cat_cols):
        encoder = cat_encoders[col]
        values = df[col].astype(str).values

        known = set(encoder.classes_)
        encoded = np.zeros(len(values), dtype=np.int64)

        for j, v in enumerate(values):
            if v in known:
                encoded[j] = encoder.transform([v])[0]
            else:
                encoded[j] = 0  # unknown token

        X_cat[:, i] = encoded

    # ----------------------------
    # TARGET
    # ----------------------------
    print("Encoding target...")
    y = target_encoder.transform(
        df[TARGET_COLUMN].astype(str).values
    ).astype(np.int64)

    # ----------------------------
    # CONVERT TO TORCH
    # ----------------------------
    print("Converting to torch tensors...")
    X_num = torch.from_numpy(X_num)
    X_cat = torch.from_numpy(X_cat)
    y = torch.from_numpy(y)

    # ----------------------------
    # SAVE CACHE
    # ----------------------------
    print("Saving tensor cache...")
    torch.save(
        (X_num, X_cat, y),
        TENSOR_CACHE_FILE
    )

    print("✅ ETL completed successfully")
    print(f"Saved to: {TENSOR_CACHE_FILE}")
    print(f"Tensor shapes:")
    print(f"  X_num: {X_num.shape}")
    print(f"  X_cat: {X_cat.shape}")
    print(f"  y:     {y.shape}")

if __name__ == "__main__":
    main()
