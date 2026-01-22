# config.py
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # DatawarehouseFinal/

TARGET_COLUMN = "dim_ocupacion_ocu_migr"

CACHE_FILE = BASE_DIR / "data_cache" / "sample_fact.parquet"
TENSOR_CACHE_FILE = BASE_DIR / "data_cache" / "train_tensor.pt"

DROP_COLS_KEYWORDS = ["id", "source_fact"]

DL_DIR = BASE_DIR / "deeplearning"

ENCODERS_DIR = DL_DIR / "encoders"