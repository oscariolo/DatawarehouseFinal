# dataset.py
import torch
from torch.utils.data import Dataset
import pandas as pd
import joblib
import numpy as np

from pathlib import Path
from config import TARGET_COLUMN

class ParquetFactDataset(Dataset):
    def __init__(self, parquet_path, chunk_size=200_000):
        self.parquet_path = parquet_path
        self.chunk_size = chunk_size

        BASE_DIR = Path(__file__).resolve().parent.parent
        ENCODER_DIR = BASE_DIR / "encoders"

        self.cat_encoders = joblib.load(ENCODER_DIR / "cat_encoders.joblib")
        self.target_encoder = joblib.load(ENCODER_DIR / "target_encoder.joblib")
        self.scaler = joblib.load(ENCODER_DIR / "scaler.joblib")
        self.num_cols = joblib.load(ENCODER_DIR / "num_cols.joblib")
        self.cat_cols = joblib.load(ENCODER_DIR / "cat_cols.joblib")


        self.df = pd.read_parquet(parquet_path)
        self.df = self.df.dropna(subset=[TARGET_COLUMN])
        self.length = len(self.df)

    @property
    def class_names(self):
        return self.target_encoder.classes_

    @property
    def n_classes(self):
        return len(self.target_encoder.classes_)
    
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ---------- NUMERIC ----------
        num_values = row[self.num_cols].values.astype(np.float32)
        num = torch.tensor(num_values)

        # ---------- CATEGORICAL (APPLY LABEL ENCODERS) ----------
        cat_values = []
        for col in self.cat_cols:
            encoder = self.cat_encoders[col]

            # unseen category safety
            value = str(row[col])
            if value in encoder.classes_:
                cat_values.append(encoder.transform([value])[0])
            else:
                cat_values.append(0)  # unknown token

        cat = torch.tensor(cat_values, dtype=torch.long)

        # ---------- TARGET ----------
        y = torch.tensor(
            self.target_encoder.transform([str(row[TARGET_COLUMN])])[0],
            dtype=torch.long
        )

        return num, cat, y
    


