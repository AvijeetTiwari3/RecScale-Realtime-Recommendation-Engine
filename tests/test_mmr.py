"""
tests/test_mmr.py
=================
Unit tests for MMR Diversification.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rec_engine.feature_store.redis_store import OnlineFeatureStore
from rec_engine.ranking.mmr_diversifier import MMRDiversifier


def test_mmr_diversification_penalty():
    store = OnlineFeatureStore()
    diversifier = MMRDiversifier(feature_store=store, lambda_param=0.5)

    # Setup candidate items with categories
    # Items 1, 2, 3 have Category "Action" with high scores [0.95, 0.94, 0.93]
    # Item 4 has Category "Comedy" with slightly lower score 0.85
    store.set_item_features(1, {"category": "Action"})
    store.set_item_features(2, {"category": "Action"})
    store.set_item_features(3, {"category": "Action"})
    store.set_item_features(4, {"category": "Comedy"})

    candidate_ids = [1, 2, 3, 4]
    scores = [0.95, 0.94, 0.93, 0.85]

    selected, selected_scores, _ = diversifier.diversify(candidate_ids, scores, top_k=2)

    assert len(selected) == 2
    assert selected[0] == 1 # First is top score
    # Due to diversity penalty on Action (1.0 similarity collision), Comedy (Item 4) must be chosen 2nd!
    assert selected[1] == 4


if __name__ == "__main__":
    test_mmr_diversification_penalty()
    print("[OK] All MMR Diversification Unit Tests Passed Successfully!")
