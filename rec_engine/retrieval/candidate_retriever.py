"""
rec_engine/retrieval/candidate_retriever.py
===========================================
Sub-5ms Candidate Retriever orchestrating User Tower & HNSW Search.
"""

import time
import torch
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

from rec_engine.models.two_tower import UserTower
from rec_engine.retrieval.hnsw_index import HNSWVectorIndex
from rec_engine.feature_store.redis_store import OnlineFeatureStore


class CandidateRetriever:
    """
    Sub-5ms Candidate Retrieval Gateway.
    Maps real-time user session signals to top-500 candidate item IDs.
    """

    def __init__(
        self,
        user_tower: UserTower,
        hnsw_index: HNSWVectorIndex,
        feature_store: OnlineFeatureStore,
        device: str = "cpu"
    ):
        self.user_tower = user_tower.to(device)
        self.user_tower.eval()
        self.hnsw_index = hnsw_index
        self.feature_store = feature_store
        self.device = device

    @torch.no_grad()
    def retrieve_candidates(self, user_id: int, top_k: int = 500) -> Tuple[List[int], List[float], float]:
        """
        Retrieves top_k candidate item IDs for user_id in sub-5ms.
        Returns: (candidate_ids, similarity_scores, latency_ms)
        """
        start_time = time.perf_counter()

        # 1. Fetch live user state from online feature store
        user_state = self.feature_store.get_user_state(user_id)
        recent_seq = user_state.recent_clicked_items[-10:] if user_state.recent_clicked_items else [0]
        
        # Pad sequence to fixed length 10
        if len(recent_seq) < 10:
            recent_seq = [0] * (10 - len(recent_seq)) + recent_seq

        # 2. Forward pass through User Tower
        u_tensor = torch.tensor([user_id], dtype=torch.long, device=self.device)
        seq_tensor = torch.tensor([recent_seq], dtype=torch.long, device=self.device)
        
        user_vec = self.user_tower(u_tensor, seq_tensor).cpu().numpy()[0]

        # 3. Query HNSW Vector Index
        cand_ids, scores = self.hnsw_index.search(user_vec, top_k=top_k)
        
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return cand_ids, scores, latency_ms
