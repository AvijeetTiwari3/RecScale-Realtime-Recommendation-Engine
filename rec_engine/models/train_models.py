"""
rec_engine/models/train_models.py
=================================
Training Pipeline for Two-Tower Retrieval & DCNv2 Ranking Models on Real MovieLens Data.
"""

import os
import sys
import csv
import time
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Ensure root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rec_engine.models.two_tower import TwoTowerModel
from rec_engine.models.dcn_v2 import DCNv2Ranker
from rec_engine.models.loss import InBatchInfoNCELoss, RankingCTRLoss


class MovieLensDataset(Dataset):
    """Parses real interaction triplets (user_id, item_id, rating/label)."""
    def __init__(self, ratings_csv_path: str, max_samples: int = 50000):
        self.samples = []
        with open(ratings_csv_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if len(row) >= 3:
                    try:
                        uid = int(row[0])
                        mid = int(row[1])
                        rating = float(row[2])
                        # Binary engagement: rating >= 4.0 is positive (1), else negative (0)
                        label = 1.0 if rating >= 4.0 else 0.0
                        self.samples.append((uid, mid, label, rating))
                        if len(self.samples) >= max_samples:
                            break
                    except Exception:
                        continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        uid, mid, label, rating = self.samples[idx]
        return {
            "user_id": torch.tensor(uid, dtype=torch.long),
            "item_id": torch.tensor(mid, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.float32),
            "rating": torch.tensor(rating, dtype=torch.float32)
        }


def train_models(
    ratings_path: str = "data_recsys/ratings.csv",
    epochs: int = 2,
    batch_size: int = 256,
    device: str = "cpu"
):
    print("=" * 70)
    print("   TRAINING REAL-TIME RECOMMENDATION & RANKING MODELS (RecScale)")
    print("=" * 70)

    dataset = MovieLensDataset(ratings_path, max_samples=25000)
    print(f"Loaded {len(dataset)} real interaction records for training.")

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # 1. Train Two-Tower Retrieval Model
    print("\n[*] Initializing & Training Two-Tower Retrieval Model with In-Batch InfoNCE...")
    two_tower = TwoTowerModel(num_users=10000, num_items=10000, embedding_dim=64, output_dim=128).to(device)
    retrieval_loss_fn = InBatchInfoNCELoss(temperature=0.07)
    optimizer_tt = torch.optim.AdamW(two_tower.parameters(), lr=1e-3, weight_decay=1e-4)

    two_tower.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in train_loader:
            u_ids = batch["user_id"].to(device)
            i_ids = batch["item_id"].to(device)
            
            # Dummy recent sequence for user tower
            seqs = torch.zeros((batch_size, 10), dtype=torch.long, device=device)
            cat_ids = torch.zeros(batch_size, dtype=torch.long, device=device)

            u_vecs = two_tower.user_tower(u_ids, seqs)
            i_vecs = two_tower.item_tower(i_ids, cat_ids)

            loss = retrieval_loss_fn(u_vecs, i_vecs)
            optimizer_tt.zero_grad()
            loss.backward()
            optimizer_tt.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f" [Two-Tower] Epoch {epoch+1}/{epochs} | InfoNCE Loss: {avg_loss:.4f}")

    # 2. Train DCNv2 Real-Time Ranker
    print("\n[*] Initializing & Training Deep & Cross Network v2 (DCNv2) for CTR Ranking...")
    dcn = DCNv2Ranker(num_users=10000, num_items=10000, embedding_dim=32, num_cross_layers=3).to(device)
    ranker_loss_fn = RankingCTRLoss()
    optimizer_dcn = torch.optim.AdamW(dcn.parameters(), lr=2e-3, weight_decay=1e-4)

    dcn.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in train_loader:
            u_ids = batch["user_id"].to(device)
            i_ids = batch["item_id"].to(device)
            labels = batch["label"].to(device)
            cat_ids = torch.zeros(batch_size, dtype=torch.long, device=device)
            dense_feats = torch.zeros((batch_size, 3), dtype=torch.float32, device=device)

            logits = dcn(u_ids, i_ids, cat_ids, dense_feats)
            loss = ranker_loss_fn(logits, labels)

            optimizer_dcn.zero_grad()
            loss.backward()
            optimizer_dcn.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f" [DCNv2 Ranker] Epoch {epoch+1}/{epochs} | Ranking BCE Loss: {avg_loss:.4f}")

    print("\n[✓] Both Retrieval & Ranking Models Trained Successfully!")
    return two_tower, dcn


if __name__ == "__main__":
    data_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data_recsys",
        "ratings.csv"
    )
    train_models(ratings_path=data_file, epochs=1, batch_size=128)
