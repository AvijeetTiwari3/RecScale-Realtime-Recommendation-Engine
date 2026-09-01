"""
tests/test_dcn_v2.py
====================
Unit tests for Deep & Cross Network v2 (DCNv2) Ranker.
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rec_engine.models.dcn_v2 import DCNv2Ranker, CrossLayerV2, CrossNetwork


def test_cross_layer_v2_formula():
    batch_size = 4
    dim = 16
    cross_layer = CrossLayerV2(input_dim=dim)

    x0 = torch.ones(batch_size, dim)
    xl = torch.ones(batch_size, dim)

    out = cross_layer(x0, xl)
    assert out.shape == (batch_size, dim)


def test_dcn_v2_ranker_forward_and_probabilities():
    batch_size = 8
    model = DCNv2Ranker(num_users=100, num_items=500, num_categories=20, embedding_dim=16)

    u_ids = torch.randint(1, 100, (batch_size,))
    i_ids = torch.randint(1, 500, (batch_size,))
    c_ids = torch.randint(1, 20, (batch_size,))
    dense = torch.randn(batch_size, 3)

    logits = model(u_ids, i_ids, c_ids, dense)
    assert logits.shape == (batch_size, 1)

    probs = model.predict_ctr(u_ids, i_ids, c_ids, dense)
    assert probs.shape == (batch_size,)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


if __name__ == "__main__":
    test_cross_layer_v2_formula()
    test_dcn_v2_ranker_forward_and_probabilities()
    print("[OK] All DCNv2 Ranking Unit Tests Passed Successfully!")
