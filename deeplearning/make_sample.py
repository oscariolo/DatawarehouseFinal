import pandas as pd
from pathlib import Path

FULL_DATA = Path("data_cache/fact_emigrante_fact_inmigrante.parquet")
SAMPLE_DATA = Path("data_cache/sample_fact.parquet")

N_ROWS = 100000  # ← tiny but enough

print("📦 Loading full dataset (metadata only)...")
df = pd.read_parquet(FULL_DATA)

print("✂️ Sampling data...")
sample = df.sample(n=N_ROWS, random_state=42)

sample.to_parquet(SAMPLE_DATA, index=False)
print(f"✅ Sample saved to {SAMPLE_DATA} ({len(sample)} rows)")
