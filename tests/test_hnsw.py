"""
tests/test_hnsw.py
==================
Unit tests for HNSW Vector Index Candidate Retrieval.
"""

import numpy as np
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rec_engine.retrieval.hnsw_index import HNSWVectorIndex


def test_hnsw_index_search():
    num_items = 200
    dim = 128
    index = HNSWVectorIndex(dimension=dim)

    # Generate synthetic unit vectors
    np.random.seed(42)
    embeddings = np.random.randn(num_items, dim).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    item_ids = list(range(1000, 1000 + num_items))

    index.build_index(item_ids, embeddings)
    assert index.total_items == num_items

    # Query with exact embedding of item_ids[5]
    query_vec = embeddings[5]
    ret_ids, scores = index.search(query_vec, top_k=10)

    assert len(ret_ids) == 10
    # Top result should be the item itself
    assert ret_ids[0] == item_ids[5]
    assert np.isclose(scores[0], 1.0, atol=1e-3)


if __name__ == "__main__":
    test_hnsw_index_search()
    print("[OK] All HNSW Vector Index Unit Tests Passed Successfully!")
