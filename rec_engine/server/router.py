"""
rec_engine/server/router.py
===========================
FastAPI Routing Endpoints for Real-Time Multi-Stage Recommendation.
"""

import time
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, Request, Response

from rec_engine.streaming.event_schemas import UserClickEvent, UserImpressionEvent
from rec_engine.server.telemetry import (
    RECSYS_TOTAL_LATENCY,
    RECSYS_RETRIEVAL_LATENCY,
    RECSYS_RANKING_LATENCY,
    RECSYS_EVENTS_INGESTED,
    RECSYS_RECOMMENDATIONS_SERVED,
    RECSYS_CACHE_HIT_RATIO,
    generate_latest,
    CONTENT_TYPE_LATEST
)

router = APIRouter()


class RecommendationRequest(BaseModel):
    user_id: int
    top_k: Optional[int] = Field(default=20, ge=1, le=100)
    retrieval_k: Optional[int] = Field(default=500, ge=10, le=1000)
    diversity_lambda: Optional[float] = Field(default=0.7, ge=0.0, le=1.0)


class RecommendedItem(BaseModel):
    item_id: int
    score: float
    category: str
    title: Optional[str] = None


class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[RecommendedItem]
    latency_breakdown_ms: Dict[str, float]
    total_candidates_retrieved: int


@router.post("/v1/recommend", response_model=RecommendationResponse)
async def get_recommendations(req: RecommendationRequest, request: Request):
    app_state = request.app.state
    if not hasattr(app_state, "pipeline") or app_state.pipeline is None:
        raise HTTPException(status_code=503, detail="Recommendation pipeline not initialized.")

    pipeline = app_state.pipeline
    start_time = time.perf_counter()

    # Stage 1 & 2: Candidate Retrieval (Sub-5ms)
    cand_ids, cand_scores, ret_lat = pipeline.retriever.retrieve_candidates(req.user_id, top_k=req.retrieval_k)
    RECSYS_RETRIEVAL_LATENCY.observe(ret_lat)

    # Stage 3: DCNv2 Ranking (Sub-10ms)
    ranked_ids, ranked_scores, rank_lat = pipeline.ranker.rank_candidates(req.user_id, cand_ids)
    RECSYS_RANKING_LATENCY.observe(rank_lat)

    # Record prediction for drift detection
    for s in ranked_scores[:10]:
        pipeline.drift_detector.record_prediction(s)

    # Stage 4: MMR Diversification (Sub-2ms)
    pipeline.diversifier.lambda_param = req.diversity_lambda
    div_ids, div_scores, mmr_lat = pipeline.diversifier.diversify(ranked_ids, ranked_scores, top_k=req.top_k)

    total_latency = (time.perf_counter() - start_time) * 1000.0
    RECSYS_TOTAL_LATENCY.observe(total_latency)
    RECSYS_RECOMMENDATIONS_SERVED.inc()
    RECSYS_CACHE_HIT_RATIO.set(pipeline.feature_store.get_hit_ratio())

    # Build response with metadata
    items = []
    for iid, score in zip(div_ids, div_scores):
        feat = pipeline.feature_store.get_item_features(iid)
        items.append(RecommendedItem(
            item_id=iid,
            score=round(score, 4),
            category=feat.get("category", "General"),
            title=feat.get("title", f"Movie #{iid}")
        ))

    return RecommendationResponse(
        user_id=req.user_id,
        recommendations=items,
        latency_breakdown_ms={
            "retrieval_ms": round(ret_lat, 2),
            "ranking_ms": round(rank_lat, 2),
            "mmr_diversity_ms": round(mmr_lat, 2),
            "end_to_end_total_ms": round(total_latency, 2)
        },
        total_candidates_retrieved=len(cand_ids)
    )


@router.post("/v1/events/click")
async def ingest_click_event(event: UserClickEvent, request: Request):
    pipeline = request.app.state.pipeline
    if pipeline:
        pipeline.aggregator.process_click_event(event)
        RECSYS_EVENTS_INGESTED.inc()
    return {"status": "accepted", "user_id": event.user_id, "item_id": event.item_id}


@router.get("/v1/drift")
async def get_drift_status(request: Request):
    pipeline = request.app.state.pipeline
    if pipeline:
        return pipeline.drift_detector.evaluate_drift()
    return {"status": "unavailable"}


@router.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.get("/healthz")
async def healthz():
    return {"status": "healthy", "service": "recscale-realtime-recommender"}
