"""
tests/test_streaming.py
=======================
Unit tests for Streaming Feature Aggregator.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rec_engine.feature_store.redis_store import OnlineFeatureStore
from rec_engine.streaming.event_schemas import UserClickEvent
from rec_engine.streaming.feature_aggregator import StreamingFeatureAggregator


def test_streaming_feature_aggregator():
    store = OnlineFeatureStore()
    aggregator = StreamingFeatureAggregator(feature_store=store, max_history_len=5)

    user_id = 42

    # Emit 3 click events
    ev1 = UserClickEvent(user_id=user_id, item_id=101, category="Sci-Fi")
    ev2 = UserClickEvent(user_id=user_id, item_id=102, category="Sci-Fi")
    ev3 = UserClickEvent(user_id=user_id, item_id=103, category="Drama")

    aggregator.process_click_event(ev1)
    aggregator.process_click_event(ev2)
    state = aggregator.process_click_event(ev3)

    assert state.user_id == user_id
    assert state.recent_clicked_items == [101, 102, 103]
    assert state.recent_categories == ["Sci-Fi", "Sci-Fi", "Drama"]
    assert state.click_count_5min == 3
    # Sci-Fi should have 2/3 affinity (~0.66), Drama 1/3 (~0.33)
    assert state.category_affinity["Sci-Fi"] > state.category_affinity["Drama"]


if __name__ == "__main__":
    test_streaming_feature_aggregator()
    print("[OK] All Streaming Feature Aggregator Unit Tests Passed Successfully!")
