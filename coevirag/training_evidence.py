"""Training-only local evidence targets; never used by graph search or inference."""

import numpy as np
import torch


def ranked_support_utility(passages, gold_passages, scores, ks=(2, 5, 10)):
    gold = set(gold_passages)
    if not gold:
        return 0.0
    ranked = sorted(set(passages).intersection(scores), key=lambda pid: (scores[pid], pid), reverse=True)
    return float(np.mean([len(set(ranked[:k]) & gold) / len(gold) for k in ks]))


def local_evidence_utilities(rag, query, nodes, gold_docs, paragraph=False):
    gold = {rag.chunk_embedding_store.text_to_hash_id[doc] for doc in gold_docs
            if doc in rag.chunk_embedding_store.text_to_hash_id}
    if not gold:
        return [0.0] * len(nodes)
    cache = getattr(rag, "_training_evidence_rank_cache", None)
    if cache is None:
        cache = rag._training_evidence_rank_cache = {}
    score_key = ("query_scores", query)
    if score_key not in cache:
        vector = rag.query_to_embedding["passage"][query]
        if isinstance(vector, torch.Tensor):
            vector = vector.detach().cpu().numpy()
        dense = np.asarray(rag.passage_embeddings @ np.asarray(vector).reshape(-1)).reshape(-1)
        cache[score_key] = dict(zip(rag.passage_node_keys, dense))
    scores = cache[score_key]
    gold_key = tuple(sorted(gold))
    values = []
    for node in nodes:
        key = ("utility", query, paragraph, gold_key, node)
        if key not in cache:
            passages = [node] if paragraph else rag.get_supporting_passages_for_node(node)
            cache[key] = ranked_support_utility(passages, gold, scores)
        values.append(cache[key])
    return values


def listwise_evidence_loss(log_probs, utilities):
    if len(utilities) != log_probs.numel():
        raise ValueError("Action targets must match the valid neighbor set")
    values = np.asarray(utilities, dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Evidence utilities must be finite and nonnegative")
    if not len(values) or values.sum() <= 0 or values.max() == values.min():
        return None
    target = torch.as_tensor(values / values.sum(), device=log_probs.device, dtype=log_probs.dtype)
    return -(target * log_probs).sum()
