"""
rec_engine/models/two_tower.py
==============================
Two-Tower Dual-Encoder PyTorch Architecture for Candidate Retrieval.

User Tower maps dynamic interaction history + static user attributes into R^d.
Item Tower maps item catalog attributes into R^d.
Vectors are L2-normalized for sub-millisecond maximum inner-product search (MIPS).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional


class UserTower(nn.Module):
    """
    Encodes user ID and sequential interaction history into a normalized user vector.
    """
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        output_dim: int = 128
    ):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users + 1, embedding_dim, padding_idx=0)
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        
        # Self-Attention sequence aggregator over recent clicks
        self.seq_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim, num_heads=2, batch_first=True
        )
        
        # Non-linear projection layers
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, output_dim)
        )

    def forward(self, user_ids: torch.Tensor, recent_item_seqs: torch.Tensor) -> torch.Tensor:
        """
        user_ids: [Batch]
        recent_item_seqs: [Batch, SeqLen]
        Returns: [Batch, OutputDim] (L2-normalized)
        """
        u_emb = self.user_embedding(user_ids) # [Batch, Dim]
        
        # Embed interaction sequence
        seq_emb = self.item_embedding(recent_item_seqs) # [Batch, SeqLen, Dim]
        
        # Self-attention pooling
        attn_out, _ = self.seq_attention(seq_emb, seq_emb, seq_emb)
        pooled_seq = torch.mean(attn_out, dim=1) # [Batch, Dim]
        
        # Concatenate static user embedding + real-time interaction vector
        combined = torch.cat([u_emb, pooled_seq], dim=-1) # [Batch, 2*Dim]
        out = self.mlp(combined) # [Batch, OutputDim]
        
        # L2-normalization for cosine / MIPS
        return F.normalize(out, p=2, dim=-1)


class ItemTower(nn.Module):
    """
    Encodes item ID and category metadata into a normalized item vector.
    """
    def __init__(
        self,
        num_items: int,
        num_categories: int = 30,
        embedding_dim: int = 64,
        output_dim: int = 128
    ):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.category_embedding = nn.Embedding(num_categories + 1, embedding_dim, padding_idx=0)

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, output_dim)
        )

    def forward(self, item_ids: torch.Tensor, category_ids: torch.Tensor) -> torch.Tensor:
        """
        item_ids: [Batch]
        category_ids: [Batch]
        Returns: [Batch, OutputDim] (L2-normalized)
        """
        i_emb = self.item_embedding(item_ids)
        c_emb = self.category_embedding(category_ids)

        combined = torch.cat([i_emb, c_emb], dim=-1)
        out = self.mlp(combined)
        return F.normalize(out, p=2, dim=-1)


class TwoTowerModel(nn.Module):
    """Dual-Encoder Candidate Retrieval Model."""
    def __init__(
        self,
        num_users: int = 10000,
        num_items: int = 10000,
        num_categories: int = 30,
        embedding_dim: int = 64,
        output_dim: int = 128
    ):
        super().__init__()
        self.user_tower = UserTower(num_users, num_items, embedding_dim, output_dim)
        self.item_tower = ItemTower(num_items, num_categories, embedding_dim, output_dim)

    def forward(
        self,
        user_ids: torch.Tensor,
        recent_item_seqs: torch.Tensor,
        item_ids: torch.Tensor,
        category_ids: torch.Tensor
    ) -> torch.Tensor:
        """Computes cosine similarity between user query and item candidate."""
        u_vec = self.user_tower(user_ids, recent_item_seqs)
        i_vec = self.item_tower(item_ids, category_ids)
        return torch.sum(u_vec * i_vec, dim=-1)
