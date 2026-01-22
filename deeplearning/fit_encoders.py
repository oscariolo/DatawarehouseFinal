import pandas as pd
from pathlib import Path
from preprocessing import fit_encoders
from config import TARGET_COLUMN, CACHE_FILE

if __name__ == "__main__":
    print("📦 Loading Parquet for encoder fitting...")
    df = pd.read_parquet(CACHE_FILE)

    # Quick safety check
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found")

    fit_encoders(df)
    print("✅ Encoder fitting completed")
