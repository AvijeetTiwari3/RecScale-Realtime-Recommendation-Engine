"""
rec_engine/ranking/mmr_diversifier.py
=====================================
Maximal Marginal Relevance (MMR) Re-Ranking for Diversity & Serendipity.

Prevents filter bubbles by balancing predicted CTR relevance with item-to-item diversity.
Formula: MMR = argmax [ lambda * Rel(u, d_i) - (1 - lambda) * max_{d_j in S} Sim(d_i, d_j) ]
"""

import time
from typing import List, Tuple, Dict, Any
from rec_engine.feature_store.redis_store import OnlineFeatureStore


class MMRDiversifier:
    """
    Greedy Maximal Marginal Relevance Re-Ranker.
    """

    def __init__(self, feature_store: OnlineFeatureStore, lambda_param: float = 0.7):
        self.feature_store = feature_store
        self.lambda_param = lambda_param

    def diversify(
        self,
        candidate_ids: List[int],
        scores: List[float],
        top_k: int = 20
    ) -> Tuple[List[int], List[float], float]:
        """
        Re-ranks scored candidates to produce a diverse top_k feed.
        Returns: (diverse_item_ids, diverse_scores, latency_ms)
        """
        start_time = time.perf_counter()

        if not candidate_ids:
            return [], [], 0.0

        item_features = {iid: self.feature_store.get_item_features(iid) for iid in candidate_ids}
        score_map = {iid: s for iid, s in zip(candidate_ids, scores)}

        selected: List[int] = []
        selected_scores: List[float] = []
        remaining = list(candidate_ids)

        while len(selected) < min(top_k, len(candidate_ids)) and remaining:
            if not selected:
                # First item is purely top relevance
                best_item = remaining.pop(0)
                selected.append(best_item)
                selected_scores.append(score_map[best_item])
            else:
                best_item = None
                best_mmr_score = -float('inf')
                best_idx = -1

                for idx, candidate in enumerate(remaining):
                    rel = score_map[candidate]
                    cand_cat = item_features[candidate].get("category", "General")

                    # Calculate max similarity to already selected items (Category-based Jaccard/Overlap)
                    max_sim = 0.0
                    for s_item in selected:
                        s_cat = item_features[s_item].get("category", "General")
                        if s_cat == cand_cat:
                            max_sim = max(max_sim, 1.0) # Category collision

                    mmr_val = self.lambda_param * rel - (1.0 - self.lambda_param) * max_sim

                    if mmr_val > best_mmr_score:
                        best_mmr_score = mmr_val
                        best_item = candidate
                        best_idx = idx

                if best_idx >= 0:
                    remaining.pop(best_idx)
                    selected.append(best_item)
                    selected_scores.append(score_map[best_item])
                else:
                    break

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return selected, selected_scores, latency_ms
