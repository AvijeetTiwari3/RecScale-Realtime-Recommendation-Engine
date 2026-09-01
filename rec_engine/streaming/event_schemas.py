"""
rec_engine/streaming/event_schemas.py
=====================================
Pydantic Schemas for Real-Time Streaming Clickstream Events.
"""

import time
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class UserClickEvent(BaseModel):
    user_id: int
    item_id: int
    timestamp: float = Field(default_factory=time.time)
    category: Optional[str] = "General"
    dwell_time_ms: Optional[int] = 0
    device: Optional[str] = "mobile"
    session_id: Optional[str] = None


class UserImpressionEvent(BaseModel):
    user_id: int
    item_ids: List[int]
    timestamp: float = Field(default_factory=time.time)
    session_id: Optional[str] = None


class UserRealtimeState(BaseModel):
    user_id: int
    recent_clicked_items: List[int] = Field(default_factory=list)
    recent_categories: List[str] = Field(default_factory=list)
    click_count_5min: int = 0
    last_event_timestamp: float = Field(default_factory=time.time)
    category_affinity: Dict[str, float] = Field(default_factory=dict)
