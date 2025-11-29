'''
Métricas de evaluación para sistemas de recomendación. Están optimizadas
para trabajar con tensores de PyTorch en GPU y batches.
'''

import numpy as np
import torch as T

def hit_rate_at_k(predictions, targets, k=10):
    """
    Hit Rate @K: proporción de veces que el item verdadero
    aparece dentro del top-k.
    """
    top_k = T.topk(predictions, k, dim=1).indices  # (batch, k)
    hits = (top_k == targets.unsqueeze(1)).any(dim=1).float()  # (batch,)
    return hits.mean().item()


def ndcg_at_k(predictions, targets, k=10):
    """
    NDCG@K: mide qué tan alto está el item verdadero en el top-k.
    """
    top_k_indices = T.topk(predictions, k, dim=1).indices  # (batch, k)
    relevance = (top_k_indices == targets.unsqueeze(1)).float()  # (batch, k)

    ranks = T.arange(1, k + 1, device=predictions.device).float()  # (k,)
    dcg = (relevance / T.log2(ranks + 1)).sum(dim=1)  # (batch,)

    idcg = 1.0 / np.log2(2)  # DCG ideal (target en posición 1)
    ndcg = dcg / idcg
    return ndcg.mean().item()


def mrr(predictions, targets):
    """
    Mean Reciprocal Rank: promedio de 1/rank del item verdadero.
    """
    sorted_indices = T.argsort(predictions, dim=1, descending=True)  # (batch, num_items)
    matches = (sorted_indices == targets.unsqueeze(1))  # (batch, num_items)
    indices = matches.nonzero(as_tuple=False)           # (batch, 2) → (batch_idx, rank_idx)
    ranks = indices[:, 1].float() + 1.0                 # rank desde 1

    rr = 1.0 / ranks
    return rr.mean().item()
