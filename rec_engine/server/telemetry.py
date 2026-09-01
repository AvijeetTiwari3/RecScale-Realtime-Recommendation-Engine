"""
rec_engine/server/telemetry.py
==============================
Prometheus Metrics Instrumentation for Real-Time Recommendation Engine.
"""

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# Latency Histograms by stage
RECSYS_TOTAL_LATENCY = Histogram(
    "recsys_end_to_end_latency_ms",
    "Total latency for end-to-end recommendation request in milliseconds",
    buckets=[2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 50.0, 100.0]
)

RECSYS_RETRIEVAL_LATENCY = Histogram(
    "recsys_retrieval_stage_latency_ms",
    "Candidate retrieval stage latency in milliseconds",
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0]
)

RECSYS_RANKING_LATENCY = Histogram(
    "recsys_ranking_stage_latency_ms",
    "DCNv2 ranking stage latency in milliseconds",
    buckets=[1.0, 2.0, 5.0, 10.0, 20.0]
)

RECSYS_EVENTS_INGESTED = Counter(
    "recsys_events_ingested_total",
    "Total count of real-time streaming clickstream events ingested"
)

RECSYS_RECOMMENDATIONS_SERVED = Counter(
    "recsys_recommendations_served_total",
    "Total count of recommendation feeds generated"
)

RECSYS_CACHE_HIT_RATIO = Gauge(
    "recsys_feature_store_cache_hit_ratio",
    "Online feature store cache hit ratio (0.0 - 1.0)"
)
