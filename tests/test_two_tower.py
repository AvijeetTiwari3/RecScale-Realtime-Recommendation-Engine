"""
tests/test_two_tower.py
=======================
Unit tests for Two-Tower Retrieval Model and InfoNCE Loss.
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rec_engine.models.two_tower import TwoTowerModel, UserTower, ItemTower
from rec_engine.models.loss import InBatchInfoNCELoss


def test_two_tower_dimensions_and_normalization():
    batch_size = 8
    seq_len = 10
    dim = 64
    out_dim = 128

    model = TwoTowerModel(num_users=100, num_items=500, embedding_dim=dim, output_dim=out_dim)

    u_ids = torch.randint(1, 100, (batch_size,))
    seqs = torch.randint(1, 500, (batch_size, seq_len))
    i_ids = torch.randint(1, 500, (batch_size,))
    cat_ids = torch.randint(1, 20, (batch_size,))

    u_vecs = model.user_tower(u_ids, seqs)
    i_vecs = model.item_tower(i_ids, cat_ids)

    assert u_vecs.shape == (batch_size, out_dim)
    assert i_vecs.shape == (batch_size, out_dim)

    # Check L2 normalization: norm should equal 1.0
    u_norms = torch.norm(u_vecs, p=2, dim=-1)
    i_norms = torch.norm(i_vecs, p=2, dim=-1)
    assert torch.allclose(u_norms, torch.ones_like(u_norms), atol=1e-5)
    assert torch.allclose(i_norms, torch.ones_like(i_norms), atol=1e-5)


def test_infonce_loss():
    batch_size = 4
    dim = 128
    loss_fn = InBatchInfoNCELoss(temperature=0.07)

    # Case 1: Perfect alignment (user vectors identical to item vectors)
    u_vecs = torch.randn(batch_size, dim)
    u_vecs = torch.nn.functional.normalize(u_vecs, p=2, dim=-1)
    i_vecs = u_vecs.clone()

    loss_aligned = loss_fn(u_vecs, i_vecs)

    # Case 2: Orthogonal / random vectors
    i_random = torch.randn(batch_size, dim)
    i_random = torch.nn.functional.normalize(i_random, p=2, dim=-1)
    loss_random = loss_fn(u_vecs, i_random)

    assert loss_aligned < loss_random


if __name__ == "__main__":
    test_two_tower_dimensions_and_normalization()
    test_infonce_loss()
    print("[OK] All Two-Tower Retrieval Unit Tests Passed Successfully!")
