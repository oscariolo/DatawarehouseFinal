# train.py
import torch
from torch.utils.data import DataLoader
from tensor_dataset import CachedTensorDataset
from model import DWEmbeddingClassifier
import joblib
from evaluate import evaluate
from config import TENSOR_CACHE_FILE, ENCODERS_DIR

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if device.type == "cuda":
        print(torch.cuda.get_device_name())

    # -------------------------------
    # LOAD PRE-ENCODED DATA (FAST)
    # -------------------------------
    dataset = CachedTensorDataset(TENSOR_CACHE_FILE)

    # -------------------------------
    # MODEL METADATA (lightweight)
    # -------------------------------
    num_cols = joblib.load(ENCODERS_DIR / "num_cols.joblib")
    cat_encoders = joblib.load(ENCODERS_DIR / "cat_encoders.joblib")
    target_encoder = joblib.load(ENCODERS_DIR / "target_encoder.joblib")

    cat_cardinalities = [
        len(enc.classes_) + 1 for enc in cat_encoders.values()
    ]

    model = DWEmbeddingClassifier(
        num_numeric=len(num_cols),
        cat_cardinalities=cat_cardinalities,
        n_classes=len(target_encoder.classes_)
    ).to(device)

    # -------------------------------
    # DATALOADER (GPU FRIENDLY)
    # -------------------------------
    loader = DataLoader(
        dataset,
        batch_size=4096,       # 🔥 increase batch size
        shuffle=True,
        num_workers=0,         # ✅ Windows-safe
        pin_memory=True,
        persistent_workers=False
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    # -------------------------------
    # TRAINING LOOP
    # -------------------------------
    for epoch in range(10):
        model.train()
        total_loss = 0.0

        for num_x, cat_x, y in loader:
            num_x = num_x.to(device, non_blocking=True)
            cat_x = cat_x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            out = model(num_x, cat_x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} | Loss: {total_loss / len(loader):.4f}")

    # -------------------------------
    # EVALUATION
    # -------------------------------
    evaluate(model, loader, device, class_names=target_encoder.classes_.tolist())

if __name__ == "__main__":
    main()
