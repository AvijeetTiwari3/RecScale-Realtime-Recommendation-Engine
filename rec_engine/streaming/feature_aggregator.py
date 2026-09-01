"""
rec_engine/streaming/feature_aggregator.py
==========================================
Streaming Feature Aggregator for Real-Time Clickstream Events.
"""

import time
from typing import List, Dict, Any, TYPE_CHECKING
from rec_engine.streaming.event_schemas import UserClickEvent, UserImpressionEvent, UserRealtimeState

if TYPE_CHECKING:
    from rec_engine.feature_store.redis_store import OnlineFeatureStore


class StreamingFeatureAggregator:
    """
    Stateful streaming processor maintaining real-time user intent profiles.
    """

    def __init__(self, feature_store: Any, max_history_len: int = 20):
        self.feature_store = feature_store
        self.max_history_len = max_history_len
        self.total_events_processed = 0

    def process_click_event(self, event: UserClickEvent) -> UserRealtimeState:
        """
        Updates the user's online state with a new real-time interaction.
        """
        self.total_events_processed += 1
        state = self.feature_store.get_user_state(event.user_id)

        # 1. Update recent item sequence
        state.recent_clicked_items.append(event.item_id)
        if len(state.recent_clicked_items) > self.max_history_len:
            state.recent_clicked_items.pop(0)

        # 2. Update category history
        category = event.category or "General"
        state.recent_categories.append(category)
        if len(state.recent_categories) > self.max_history_len:
            state.recent_categories.pop(0)

        # 3. Calculate category affinity distribution from sliding window
        cat_counts: Dict[str, float] = {}
        for cat in state.recent_categories:
            cat_counts[cat] = cat_counts.get(cat, 0.0) + 1.0

        total = float(len(state.recent_categories))
        state.category_affinity = {cat: count / total for cat, count in cat_counts.items()}

        # 4. Update velocity and timestamps
        now = event.timestamp or time.time()
        time_diff = now - state.last_event_timestamp
        if time_diff < 300: # Within 5 minutes
            state.click_count_5min += 1
        else:
            state.click_count_5min = 1

        state.last_event_timestamp = now
        self.feature_store.save_user_state(state)
        return state
