"""
rec_engine/pipeline.py
======================
End-to-End Real-Time Recommendation Pipeline Orchestrator.
"""

import os
import csv
import torch
import numpy as np
from typing import Optional, Dict, Any, List

from rec_engine.feature_store.redis_store import OnlineFeatureStore
from rec_engine.streaming.feature_aggregator import StreamingFeatureAggregator
from rec_engine.models.two_tower import TwoTowerModel
from rec_engine.models.dcn_v2 import DCNv2Ranker
from rec_engine.retrieval.hnsw_index import HNSWVectorIndex
from rec_engine.retrieval.candidate_retriever import CandidateRetriever
from rec_engine.ranking.dcn_ranker import RealtimeRanker
from rec_engine.ranking.mmr_diversifier import MMRDiversifier
from rec_engine.monitoring.drift_detector import DriftDetector


class RecommendationPipeline:
    """
    Coordinates Candidate Retrieval, Online Feature Joining, DCNv2 Ranking, and MMR Diversity.
    """

    def __init__(
        self,
        num_users: int = 10000,
        num_items: int = 10000,
        embedding_dim: int = 64,
        device: str = "cpu"
    ):
        self.device = device
        self.num_users = num_users
        self.num_items = num_items

        # 1. Online Feature Store & Streaming Aggregator
        self.feature_store = OnlineFeatureStore()
        self.aggregator = StreamingFeatureAggregator(self.feature_store)

        # 2. Retrieval Models & HNSW Index
        self.two_tower = TwoTowerModel(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=embedding_dim,
            output_dim=128
        ).to(device)
        self.hnsw_index = HNSWVectorIndex(dimension=128)
        self.retriever = CandidateRetriever(
            user_tower=self.two_tower.user_tower,
            hnsw_index=self.hnsw_index,
            feature_store=self.feature_store,
            device=device
        )

        # 3. Real-Time Ranker (DCNv2) & MMR Diversifier
        self.dcn_model = DCNv2Ranker(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=32,
            num_cross_layers=3
        ).to(device)
        self.ranker = RealtimeRanker(
            dcn_model=self.dcn_model,
            feature_store=self.feature_store,
            device=device
        )
        self.diversifier = MMRDiversifier(feature_store=self.feature_store)

        # 4. Drift Detector
        self.drift_detector = DriftDetector()

    def bootstrap_catalog(self, movies_csv_path: str, max_items: int = 4000):
        """Populates item catalog metadata and initializes HNSW vector index."""
        if not os.path.exists(movies_csv_path):
            print(f"[!] Catalog not found at {movies_csv_path}. Initializing fallback catalog.")
            item_ids = list(range(1, 501))
            for iid in item_ids:
                self.feature_store.set_item_features(iid, {
                    "item_id": iid,
                    "title": f"Item #{iid}",
                    "category": "Action" if iid % 2 == 0 else "Drama",
                    "popularity": round(float(np.random.exponential(2.0)), 2)
                })
            vecs = np.random.randn(len(item_ids), 128).astype(np.float32)
            vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
            self.hnsw_index.build_index(item_ids, vecs)
            return

        item_ids = []
        with open(movies_csv_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    parts = line.strip().split(",")
                if len(parts) >= 3:
                    try:
                        # Handle either [idx, movie_id, title, genres] or [movie_id, title, genres]
                        if parts[0].isdigit() and len(parts) >= 4 and parts[1].isdigit():
                            mid = int(parts[1])
                            title = parts[2]
                            genres = parts[3].split("|")[0] if parts[3] else "General"
                        elif parts[0].isdigit():
                            mid = int(parts[0])
                            title = parts[1]
                            genres = parts[2].split("|")[0] if parts[2] else "General"
                        else:
                            continue

                        self.feature_store.set_item_features(mid, {
                            "item_id": mid,
                            "title": title,
                            "category": genres,
                            "popularity": round(float(np.random.exponential(1.5)), 2)
                        })
                        item_ids.append(mid)
                        if len(item_ids) >= max_items:
                            break
                    except Exception:
                        continue

        if not item_ids:
            item_ids = list(range(1, 501))
            for iid in item_ids:
                self.feature_store.set_item_features(iid, {"item_id": iid, "category": "Action", "popularity": 1.0})

        with torch.no_grad():
            i_tensors = torch.tensor(item_ids, dtype=torch.long, device=self.device)
            c_tensors = torch.zeros(len(item_ids), dtype=torch.long, device=self.device)
            item_vecs = self.two_tower.item_tower(i_tensors, c_tensors).cpu().numpy()

        self.hnsw_index.build_index(item_ids, item_vecs)
        print(f"[Pipeline] Loaded {len(item_ids)} catalog items and initialized HNSW Vector Index.")
