"""
rec_engine/ranking/dcn_ranker.py
================================
Sub-10ms Batched Real-Time Ranker using Deep & Cross Network v2.
"""

import time
import torch
from typing import List, Tuple, Dict, Any, Optional

from rec_engine.models.dcn_v2 import DCNv2Ranker
from rec_engine.feature_store.redis_store import OnlineFeatureStore


class RealtimeRanker:
    """
    Batched real-time candidate scoring engine.
    Scores 500 candidates in <10ms using DCNv2.
    """

    def __init__(
        self,
        dcn_model: DCNv2Ranker,
        feature_store: OnlineFeatureStore,
        category_map: Optional[Dict[str, int]] = None,
        device: str = "cpu"
    ):
        self.dcn_model = dcn_model.to(device)
        self.dcn_model.eval()
        self.feature_store = feature_store
        self.category_map = category_map or {"Action": 1, "Comedy": 2, "Drama": 3, "Thriller": 4, "General": 5}
        self.device = device

    @torch.no_grad()
    def rank_candidates(
        self,
        user_id: int,
        candidate_ids: List[int]
    ) -> Tuple[List[int], List[float], float]:
        """
        Enriches and scores candidate items using DCNv2.
        Returns: (sorted_candidate_ids, predicted_ctr_scores, latency_ms)
        """
        start_time = time.perf_counter()

        if not candidate_ids:
            return [], [], 0.0

        batch_size = len(candidate_ids)
        user_state = self.feature_store.get_user_state(user_id)

        # 1. Feature Enrichment
        item_features = self.feature_store.batch_get_item_features(candidate_ids)

        u_tensor = torch.full((batch_size,), user_id, dtype=torch.long, device=self.device)
        i_tensor = torch.tensor(candidate_ids, dtype=torch.long, device=self.device)

        cat_ids = []
        dense_rows = []

        for feat in item_features:
            cat_name = feat.get("category", "General")
            c_id = self.category_map.get(cat_name, 1)
            cat_ids.append(c_id)

            # Dense features: [click_count_5min, popularity, affinity_to_this_cat]
            pop = feat.get("popularity", 1.0)
            affinity = user_state.category_affinity.get(cat_name, 0.0)
            dense_rows.append([user_state.click_count_5min, pop, affinity])

        c_tensor = torch.tensor(cat_ids, dtype=torch.long, device=self.device)
        dense_tensor = torch.tensor(dense_rows, dtype=torch.float32, device=self.device)

        # 2. Batched DCNv2 Forward Pass
        predicted_probs = self.dcn_model.predict_ctr(u_tensor, i_tensor, c_tensor, dense_tensor)
        scores = predicted_probs.cpu().numpy()

        # 3. Sort candidates by predicted CTR
        sorted_indices = scores.argsort()[::-1]
        sorted_ids = [candidate_ids[idx] for idx in sorted_indices]
        sorted_scores = [float(scores[idx]) for idx in sorted_indices]

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return sorted_ids, sorted_scores, latency_ms
