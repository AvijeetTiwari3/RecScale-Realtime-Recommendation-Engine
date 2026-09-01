"""
rec_engine/models/dcn_v2.py
===========================
Deep & Cross Network v2 (DCNv2) for High-Precision Real-Time Ranking.

Models explicit bounded-degree polynomial feature interactions via Cross Layers
combined with deep non-linear multi-layer perceptron for CTR prediction.
Reference: Wang et al., Google Research, 2021 (DCN-V2).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class CrossLayerV2(nn.Module):
    """
    Single Cross Layer in DCNv2.
    Formula: x_{l+1} = x_0 * (W_l * x_l + b_l) + x_l
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, input_dim, bias=True)

    def forward(self, x0: torch.Tensor, xl: torch.Tensor) -> torch.Tensor:
        """
        x0: Initial input feature vector [Batch, Dim]
        xl: Output of the previous cross layer [Batch, Dim]
        """
        return x0 * self.linear(xl) + xl


class CrossNetwork(nn.Module):
    """Stack of multiple Cross Layers."""
    def __init__(self, input_dim: int, num_layers: int = 3):
        super().__init__()
        self.layers = nn.ModuleList([CrossLayerV2(input_dim) for _ in range(num_layers)])

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        xl = x0
        for layer in self.layers:
            xl = layer(x0, xl)
        return xl


class DCNv2Ranker(nn.Module):
    """
    Stacked & Parallel Deep & Cross Network v2 for Real-Time CTR / Engagement Prediction.
    """
    def __init__(
        self,
        num_users: int = 10000,
        num_items: int = 10000,
        num_categories: int = 30,
        embedding_dim: int = 32,
        num_cross_layers: int = 3,
        deep_layer_dims: List[int] = [128, 64, 32]
    ):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users + 1, embedding_dim, padding_idx=0)
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.cat_embedding = nn.Embedding(num_categories + 1, embedding_dim, padding_idx=0)

        # Dense numerical features: (user_5min_velocity, dwell_time, popularity) = 3 features
        num_dense_features = 3
        total_input_dim = (embedding_dim * 3) + num_dense_features

        # 1. Cross Network
        self.cross_network = CrossNetwork(input_dim=total_input_dim, num_layers=num_cross_layers)

        # 2. Deep Network
        deep_layers = []
        prev_dim = total_input_dim
        for dim in deep_layer_dims:
            deep_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = dim
        self.deep_network = nn.Sequential(*deep_layers)

        # 3. Combination & Output Head
        combined_dim = total_input_dim + deep_layer_dims[-1]
        self.output_head = nn.Linear(combined_dim, 1)

    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        cat_ids: torch.Tensor,
        dense_features: torch.Tensor
    ) -> torch.Tensor:
        """
        user_ids: [Batch]
        item_ids: [Batch]
        cat_ids: [Batch]
        dense_features: [Batch, 3]
        Returns: logits of shape [Batch, 1]
        """
        u_emb = self.user_embedding(user_ids)
        i_emb = self.item_embedding(item_ids)
        c_emb = self.cat_embedding(cat_ids)

        x0 = torch.cat([u_emb, i_emb, c_emb, dense_features], dim=-1)

        cross_out = self.cross_network(x0)
        deep_out = self.deep_network(x0)

        combined = torch.cat([cross_out, deep_out], dim=-1)
        logits = self.output_head(combined)
        return logits

    def predict_ctr(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        cat_ids: torch.Tensor,
        dense_features: torch.Tensor
    ) -> torch.Tensor:
        """Outputs calibrated probabilities in [0.0, 1.0]."""
        with torch.no_grad():
            logits = self.forward(user_ids, item_ids, cat_ids, dense_features)
            return torch.sigmoid(logits).view(-1)
