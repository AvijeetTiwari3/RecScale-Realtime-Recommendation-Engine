"""
rec_engine/feature_store/redis_store.py
=======================================
Low-Latency Online Feature Store (In-Memory / Redis Interface).

Provides sub-millisecond retrieval of real-time user sliding-window features
and static item catalog attributes.
"""

import time
import json
from typing import Dict, Any, Optional, List
from rec_engine.streaming.event_schemas import UserRealtimeState


class OnlineFeatureStore:
    """
    High-speed key-value online feature store with microsecond lookup latency.
    Supports in-memory dictionary caching and optional live Redis cluster connection.
    """

    def __init__(self, redis_host: Optional[str] = None, redis_port: int = 6379):
        self.redis_client = None
        self._user_store: Dict[int, UserRealtimeState] = {}
        self._item_store: Dict[int, Dict[str, Any]] = {}
        self.total_lookups = 0
        self.cache_hits = 0

        if redis_host:
            try:
                import redis
                self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
                self.redis_client.ping()
                print(f"[FeatureStore] Connected to Redis at {redis_host}:{redis_port}")
            except Exception as e:
                print(f"[FeatureStore] Redis connection unavailable ({e}). Using in-memory store.")
                self.redis_client = None

    def get_user_state(self, user_id: int) -> UserRealtimeState:
        """Retrieves live user state in sub-millisecond time."""
        self.total_lookups += 1
        if self.redis_client:
            try:
                data = self.redis_client.get(f"user_state:{user_id}")
                if data:
                    self.cache_hits += 1
                    return UserRealtimeState.model_validate_json(data)
            except Exception:
                pass

        if user_id in self._user_store:
            self.cache_hits += 1
            return self._user_store[user_id]
        
        # Default cold-start user state
        default_state = UserRealtimeState(user_id=user_id)
        self._user_store[user_id] = default_state
        return default_state

    def save_user_state(self, state: UserRealtimeState):
        """Persists updated user state."""
        self._user_store[state.user_id] = state
        if self.redis_client:
            try:
                self.redis_client.set(
                    f"user_state:{state.user_id}",
                    state.model_dump_json(),
                    ex=86400 # 24 hour TTL
                )
            except Exception:
                pass

    def get_item_features(self, item_id: int) -> Dict[str, Any]:
        """Retrieves item metadata (category, popularity, price tier)."""
        return self._item_store.get(item_id, {
            "item_id": item_id,
            "category": "General",
            "popularity": 1.0,
            "genres": ["Action"]
        })

    def set_item_features(self, item_id: int, features: Dict[str, Any]):
        self._item_store[item_id] = features

    def batch_get_item_features(self, item_ids: List[int]) -> List[Dict[str, Any]]:
        """Batched item feature retrieval."""
        return [self.get_item_features(iid) for iid in item_ids]

    def get_hit_ratio(self) -> float:
        return self.cache_hits / self.total_lookups if self.total_lookups > 0 else 1.0
