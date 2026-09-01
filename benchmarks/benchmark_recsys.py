"""
benchmarks/benchmark_recsys.py
==============================
End-to-End Latency Breakdown & Retrieval Recall Benchmark.
"""

import os
import sys
import time
import numpy as np

# Ensure root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rec_engine.pipeline import RecommendationPipeline
from rec_engine.streaming.event_schemas import UserClickEvent


def run_recsys_benchmark(num_users: int = 100):
    print("=" * 70)
    print("   RecScale: REAL-TIME EVENT-DRIVEN RECOMMENDATION BENCHMARK")
    print("=" * 70)

    pipeline = RecommendationPipeline()
    catalog_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_recsys", "movies.csv")
    pipeline.bootstrap_catalog(catalog_path)

    # 1. Warm-up and simulate streaming click events
    print("\n[*] Simulating real-time clickstream ingestion...")
    for uid in range(1, num_users + 1):
        for _ in range(3):
            item_id = int(np.random.randint(1, 300))
            event = UserClickEvent(user_id=uid, item_id=item_id, category="Action" if item_id % 2 == 0 else "Drama")
            pipeline.aggregator.process_click_event(event)

    print(f"[OK] Ingested streaming events across {num_users} active sessions.")

    # 2. Benchmark Multi-Stage Recommendation Pipeline
    retrieval_latencies = []
    ranking_latencies = []
    mmr_latencies = []
    total_latencies = []

    print("\n" + "-" * 70)
    print(f"{'User ID':<10} | {'Candidates':<12} | {'Retrieval (ms)':<15} | {'Ranking (ms)':<14} | {'Total (ms)':<12}")
    print("-" * 70)

    for uid in range(1, min(15, num_users + 1)):
        t0 = time.perf_counter()

        # Stage 1: Retrieval (Top-500)
        cands, _, ret_lat = pipeline.retriever.retrieve_candidates(uid, top_k=500)
        
        # Stage 2: DCNv2 Ranking
        ranked_ids, scores, rank_lat = pipeline.ranker.rank_candidates(uid, cands)
        
        # Stage 3: MMR Diversification (Top-20)
        div_ids, _, mmr_lat = pipeline.diversifier.diversify(ranked_ids, scores, top_k=20)
        
        tot_lat = (time.perf_counter() - t0) * 1000.0

        retrieval_latencies.append(ret_lat)
        ranking_latencies.append(rank_lat)
        mmr_latencies.append(mmr_lat)
        total_latencies.append(tot_lat)

        print(f"User #{uid:<5} | {len(cands):<12} | {ret_lat:<15.2f} | {rank_lat:<14.2f} | {tot_lat:<12.2f}")

    for uid in range(15, num_users + 1):
        t0 = time.perf_counter()
        cands, _, ret_lat = pipeline.retriever.retrieve_candidates(uid, top_k=500)
        ranked_ids, scores, rank_lat = pipeline.ranker.rank_candidates(uid, cands)
        div_ids, _, mmr_lat = pipeline.diversifier.diversify(ranked_ids, scores, top_k=20)
        tot_lat = (time.perf_counter() - t0) * 1000.0

        retrieval_latencies.append(ret_lat)
        ranking_latencies.append(rank_lat)
        mmr_latencies.append(mmr_lat)
        total_latencies.append(tot_lat)

    p50_total = np.percentile(total_latencies, 50)
    p95_total = np.percentile(total_latencies, 95)
    p99_total = np.percentile(total_latencies, 99)

    print("-" * 70)
    print("\n" + "=" * 70)
    print("                 FINAL LATENCY PROFILE (500 Candidates -> Top-20)")
    print("=" * 70)
    print(f" * Average Candidate Retrieval Stage : {np.mean(retrieval_latencies):.2f} ms")
    print(f" * Average DCNv2 Ranking Stage        : {np.mean(ranking_latencies):.2f} ms")
    print(f" * Average MMR Diversification Stage  : {np.mean(mmr_latencies):.2f} ms")
    print(f" * End-to-End P50 Latency             : {p50_total:.2f} ms")
    print(f" * End-to-End P95 Latency             : {p95_total:.2f} ms")
    print(f" * End-to-End P99 Latency (SLA Target): {p99_total:.2f} ms (SLA Budget < 25.0 ms)")
    print("=" * 70)


if __name__ == "__main__":
    run_recsys_benchmark(num_users=100)
