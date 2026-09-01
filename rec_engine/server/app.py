"""
rec_engine/server/app.py
========================
FastAPI Application Entry Point for Real-Time Recommendation Microservice.
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from rec_engine.pipeline import RecommendationPipeline
from rec_engine.server.router import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[*] Initializing Real-Time Recommendation Pipeline...")
    pipeline = RecommendationPipeline()
    
    # Load catalog metadata from data_recsys
    catalog_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data_recsys",
        "movies.csv"
    )
    pipeline.bootstrap_catalog(catalog_path)
    
    app.state.pipeline = pipeline
    print("[✓] Recommendation Engine ready to serve requests.")
    yield
    print("[*] Shutting down Recommendation Service...")


def create_app() -> FastAPI:
    app = FastAPI(
        title="RecScale: Real-Time Event-Driven Recommendation & Ranking Engine",
        description="Production-grade multi-stage recommendation service featuring Two-Tower Retrieval, HNSW Vector Index, DCNv2 Ranking, and Online Feature Store.",
        version="1.0.0",
        lifespan=lifespan
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)
    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("rec_engine.server.app:app", host="0.0.0.0", port=8001, reload=False)
