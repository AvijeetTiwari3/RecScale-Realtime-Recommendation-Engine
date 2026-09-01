"""
rec_engine/models/loss.py
=========================
Loss Functions for Neural Retrieval & CTR Ranking.

Implements In-Batch Negative InfoNCE Loss with Temperature Scaling
and Binary Cross-Entropy for Deep Ranking.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InBatchInfoNCELoss(nn.Module):
    """
    InfoNCE Loss with In-Batch Negative Sampling (Dual-Encoder Retrieval).
    
    Treats off-diagonal elements in the batch as negative candidates,
    avoiding the computational cost of computing softmax over millions of catalog items.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, user_embeddings: torch.Tensor, item_embeddings: torch.Tensor) -> torch.Tensor:
        """
        user_embeddings: [Batch, Dim] (L2-normalized)
        item_embeddings: [Batch, Dim] (L2-normalized)
        """
        # Similarity matrix: [Batch, Batch]
        logits = torch.matmul(user_embeddings, item_embeddings.T) / self.temperature
        
        # Ground-truth labels: diagonal indices [0, 1, 2, ..., Batch-1]
        labels = torch.arange(user_embeddings.size(0), device=user_embeddings.device)
        
        loss = self.cross_entropy(logits, labels)
        return loss


class RankingCTRLoss(nn.Module):
    """Binary Cross Entropy Loss with optional positive class weighting."""
    def __init__(self, pos_weight: float = 1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce(logits.view(-1), targets.view(-1).float())
