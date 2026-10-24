"""Dense anchoring, evidence-set selection, and recall summaries."""
import json
from pathlib import Path
import numpy as np
import torch
from . import RECALL_KS
from .features import conditional_features, normalized_unary_prior
from .models import candidate_unary_scores
from ..query_plan_evidence import (
    append_selection_progress, dynamic_query_plan_enabled, query_plan_slot_scores,
    query_plan_state_features, mask_query_plan_features,
)


def query_budgets(split_path, query_cache_path, query_ids):
    samples = json.loads(Path(split_path).read_text(encoding="utf-8"))
    cache = json.loads(Path(query_cache_path).read_text(encoding="utf-8"))
    budgets = {}
    for query_id in np.unique(query_ids):
        question = samples[int(query_id)]["question"]
        triple_count = len(cache.get(question, {}).get("raw_triples", []))
        budgets[int(query_id)] = min(4, max(2, triple_count))
    return budgets


def select_ranking(
        base, passage_ids, scores, model, checkpoint, mode, budget, metadata,
        device, tail_scores=None, raw_query_triples=None,
        query_plan_scores=None):
    order = np.argsort(-scores, kind="stable")
    tail_order = (
        order if tail_scores is None
        else np.argsort(-tail_scores, kind="stable")
    )
    selected = [int(order[0])]
    eligible = np.zeros(len(order), dtype=bool)
    limit = int(checkpoint.get("bridge_candidate_limit", 0))
    eligible[order[:limit if limit > 0 else len(order)]] = True
    mean = checkpoint["feature_mean"].cpu().numpy()
    std = checkpoint["feature_std"].cpu().numpy()
    selector_mode = str(checkpoint.get("selector_mode", "residual"))
    dynamic_plan = dynamic_query_plan_enabled(checkpoint)
    plan_cache = {}
    plan_scores = query_plan_scores if query_plan_scores is not None else (
        query_plan_slot_scores(passage_ids, metadata, raw_query_triples)
        if checkpoint.get("uses_query_plan_features", False) else None
    )

    while len(selected) < budget:
        candidate_mask = eligible.copy()
        candidate_mask[selected] = False
        candidate_indices = np.flatnonzero(candidate_mask)
        if not len(candidate_indices):
            break
        references = selected[-1:] if mode == "last" else selected
        reference_scores = []
        plan_features = (query_plan_state_features(
            passage_ids, metadata, raw_query_triples or [], selected,
            dynamic=dynamic_plan, static_scores=plan_scores, cache=plan_cache,
        ) if checkpoint.get("uses_query_plan_features", False) else None)
        for reference in references:
            features = conditional_features(
                base, scores, reference,
                passage_ids=passage_ids,
                openie_metadata=metadata,
            )
            if plan_features is not None:
                features = np.concatenate((features, plan_features), axis=1)
            if checkpoint.get("uses_selection_progress_features", False):
                features = append_selection_progress(features, len(selected), budget)
            normalized = (features[candidate_indices] - mean) / std
            normalized = mask_query_plan_features(normalized, checkpoint)
            with torch.no_grad():
                logits = model(
                    torch.from_numpy(normalized).float().to(device)
                ).view(-1).cpu().numpy()
            reference_scores.append(logits)
        bridge_scores = np.max(reference_scores, axis=0)
        if selector_mode == "residual":
            bridge_scores += normalized_unary_prior(scores, candidate_mask)
        selected.append(int(candidate_indices[int(np.argmax(bridge_scores))]))

    selected_set = set(selected)
    selected.extend(
        int(idx) for idx in tail_order if int(idx) not in selected_set
    )
    return np.asarray(selected, dtype=np.int64)


def summarize(values):
    values = np.asarray(values, dtype=np.float32)
    return {
        "metrics": {
            f"Recall@{k}": float(values[:, index].mean())
            for index, k in enumerate(RECALL_KS)
        },
        "mean_recall_2_5_10": float(values[:, (1, 2, 3)].mean()),
        "per_query_recall": values.tolist(),
    }


def prepare_groups(
        data, unary_model, unary_checkpoint, device, openie_metadata=None,
        bridge_candidate_limit=0, unary_mode="learned", query_plans=None):
    groups = []
    for query_id in np.unique(data["query_ids"]):
        indices = np.flatnonzero(data["query_ids"] == query_id)
        scores = candidate_unary_scores(
            data, indices, unary_mode, unary_model, unary_checkpoint, device
        )
        order = np.argsort(-scores, kind="stable")
        anchor = int(order[0])
        eligible = np.ones(len(indices), dtype=bool)
        if bridge_candidate_limit > 0:
            eligible[:] = False
            eligible[order[:bridge_candidate_limit]] = True
        passage_ids = (
            data["passage_ids"][indices]
            if data.get("passage_ids") is not None else None
        )
        pair_features = conditional_features(
            data["base"][indices], scores, anchor,
            passage_ids=passage_ids,
            openie_metadata=openie_metadata,
            raw_query_triples=(
                query_plans.get(int(query_id), []) if query_plans is not None else None
            ),
            selected_indices=[anchor],
        )
        groups.append({
            "query_id": int(query_id),
            "indices": indices,
            "scores": scores,
            "order": order,
            "anchor": anchor,
            "eligible": eligible,
            "pair_features": pair_features,
            "labels": data["labels"][indices],
        })
    return groups
