# model.py
import torch
import torch.nn as nn

class DWEmbeddingClassifier(nn.Module):
    def __init__(self, num_numeric, cat_cardinalities, emb_dim=16, n_classes=10):
        super().__init__()

        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinality, emb_dim)
            for cardinality in cat_cardinalities
        ])

        total_input = num_numeric + emb_dim * len(cat_cardinalities)

        self.classifier = nn.Sequential(
            nn.Linear(total_input, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes)
        )

    def forward(self, num_x, cat_x):
        emb = [
            emb_layer(cat_x[:, i])
            for i, emb_layer in enumerate(self.embeddings)
        ]
        emb = torch.cat(emb, dim=1)
        x = torch.cat([num_x, emb], dim=1)
        return self.classifier(x)
