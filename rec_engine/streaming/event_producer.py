"""
rec_engine/streaming/event_producer.py
======================================
Real-Time Clickstream Event Producer & Replayer.
"""

import os
import time
import csv
from typing import Generator, Optional, Any
from rec_engine.streaming.event_schemas import UserClickEvent


class ClickstreamEventProducer:
    """
    High-throughput event producer streaming real human click events.
    """

    def __init__(self, ratings_csv_path: str, movies_csv_path: Optional[str] = None):
        self.ratings_csv_path = ratings_csv_path
        self.movie_genres = {}

        if movies_csv_path and os.path.exists(movies_csv_path):
            with open(movies_csv_path, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                for row in reader:
                    if len(row) >= 3:
                        try:
                            mid = int(row[0])
                            genres = row[2].split("|")[0] if row[2] else "General"
                            self.movie_genres[mid] = genres
                        except Exception:
                            continue

    def stream_events(self, max_events: Optional[int] = 1000) -> Generator[UserClickEvent, None, None]:
        """Streams real click events parsed from MovieLens interactions."""
        if not os.path.exists(self.ratings_csv_path):
            raise FileNotFoundError(f"Ratings CSV not found at {self.ratings_csv_path}")

        count = 0
        with open(self.ratings_csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if len(row) >= 3:
                    try:
                        uid = int(row[0])
                        mid = int(row[1])
                        rating = float(row[2])
                        if rating >= 3.0:
                            category = self.movie_genres.get(mid, "General")
                            event = UserClickEvent(
                                user_id=uid,
                                item_id=mid,
                                category=category,
                                timestamp=time.time(),
                                dwell_time_ms=int(rating * 1000)
                            )
                            yield event
                            count += 1
                            if max_events and count >= max_events:
                                break
                    except Exception:
                        continue

    def replay_into_aggregator(self, aggregator: Any, num_events: int = 500) -> int:
        """Replays real events directly into the streaming aggregator."""
        ingested = 0
        for event in self.stream_events(max_events=num_events):
            aggregator.process_click_event(event)
            ingested += 1
        return ingested
