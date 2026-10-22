# Modified for CoEviRAG: release scope, portable paths, or fixed-protocol guards.
"""QueryPlan coverage features shared by selector training and retrieval."""

import re
from typing import Dict, Iterable, List, Sequence

import numpy as np


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
UNKNOWN_VALUES = {"", "?", "unknown", "none", "null"}
QUERY_PLAN_FEATURE_DIM = 12
SELECTION_PROGRESS_DIM = 2


def mask_query_plan_features(normalized, checkpoint):
    if checkpoint.get("ablated_query_plan_features", False):
        raise ValueError("CoEviRAG requires fixed RQR features")
    return normalized


def dynamic_query_plan_enabled(checkpoint):
    if checkpoint.get("uses_dynamic_query_plan", False) or checkpoint.get("query_plan_binding_schema"):
        raise ValueError("Dynamic query plans are excluded from CoEviRAG")
    return False


def valid_openie_triples(triples: Iterable) -> list[tuple[str, str, str]]:
    return [tuple(triple[:3]) for triple in triples
            if isinstance(triple, (tuple, list)) and len(triple) >= 3
            and all(isinstance(value, str) and _known(value) and _phrase(value)
                    for value in triple[:3])]








def query_plan_state_features(passage_ids, metadata, raw_triples, selected_indices, *,
                              dynamic=False, static_scores=None, cache=None, trace=None):
    if dynamic:
        raise ValueError("CoEviRAG uses a fixed relational query representation")
    scores = static_scores if static_scores is not None else query_plan_slot_scores(
        passage_ids, metadata, raw_triples)
    return query_plan_features_from_scores(scores, selected_indices)


def selection_progress_values(selected_count: int, budget: int) -> np.ndarray:
    if int(budget) != budget or int(selected_count) != selected_count:
        raise ValueError("Selection count and budget must be integers")
    if budget < 1 or selected_count < 0 or selected_count > budget:
        raise ValueError("Selection count must lie within a positive budget")
    return np.asarray((selected_count, budget - selected_count), dtype=np.float32)


def append_selection_progress(features: np.ndarray, selected_count: int,
                              budget: int) -> np.ndarray:
    values = selection_progress_values(selected_count, budget)
    return np.concatenate((features, np.broadcast_to(values, (*features.shape[:-1],
                                                              SELECTION_PROGRESS_DIM))), axis=-1)


def _tokens(value: object) -> set[str]:
    return set(TOKEN_PATTERN.findall(str(value).lower()))


def _phrase(value: object) -> str:
    return " ".join(TOKEN_PATTERN.findall(str(value).lower()))


def _known(value: object) -> bool:
    return str(value).strip().lower() not in UNKNOWN_VALUES and not str(value).strip().startswith("?")


def _soft_match(query_tokens: set[str], evidence_tokens: Iterable[set[str]]) -> float:
    if not query_tokens:
        return 0.0
    best = 0.0
    for candidate in evidence_tokens:
        if not candidate:
            continue
        overlap = len(query_tokens & candidate)
        best = max(best, overlap / max(1, len(query_tokens)))
    return best


def _slot_scores(metadata: Dict, raw_triples: Sequence[Sequence[object]]) -> np.ndarray:
    text = str(metadata.get("text", "")).lower()
    entity_sets = [_tokens(value) for value in metadata.get("entities", set())]
    relation_sets = [_tokens(value) for value in metadata.get("relations", set())]
    rows: List[tuple[float, float, float]] = []
    for triple in raw_triples:
        if not isinstance(triple, (list, tuple)) or len(triple) < 3:
            continue
        subject, relation, obj = triple[:3]
        known_entities = [
            value for value in (subject, obj) if _known(value)
        ]
        entity_score = 0.0
        for value in known_entities:
            phrase = _phrase(value)
            tokens = _tokens(value)
            entity_score = max(
                entity_score,
                float(bool(phrase) and phrase in text),
                _soft_match(tokens, entity_sets),
            )
        relation_phrase = _phrase(relation)
        relation_tokens = _tokens(relation)
        relation_score = max(
            float(bool(relation_phrase) and relation_phrase in text),
            _soft_match(relation_tokens, relation_sets),
        )
        # A relation-only slot remains useful when both endpoints are variables.
        if known_entities:
            joint_score = 0.5 * entity_score + 0.5 * relation_score
        else:
            joint_score = relation_score
        rows.append((entity_score, relation_score, joint_score))
    return np.asarray(rows, dtype=np.float32).reshape(-1, 3)


def query_plan_slot_scores(
        passage_ids: Sequence[object], metadata: Dict[str, Dict],
        raw_triples: Sequence[Sequence[object]]) -> np.ndarray:
    """Precompute passage-to-slot entity, relation, and joint matches."""
    if not raw_triples:
        return np.zeros((len(passage_ids), 0, 3), dtype=np.float32)
    scores = [
        _slot_scores(metadata.get(str(passage_id), {}), raw_triples)
        for passage_id in passage_ids
    ]
    slot_count = max((len(item) for item in scores), default=0)
    if slot_count == 0:
        return np.zeros((len(passage_ids), 0, 3), dtype=np.float32)
    return np.asarray([
        item if len(item) == slot_count else np.zeros((slot_count, 3), dtype=np.float32)
        for item in scores
    ], dtype=np.float32)


def query_plan_features_from_scores(
        scores: np.ndarray, selected_indices: Sequence[int]) -> np.ndarray:
    """Build dynamic coverage features from a precomputed slot-match tensor."""
    count = len(scores)
    if scores.ndim != 3 or scores.shape[1] == 0:
        return np.zeros((count, QUERY_PLAN_FEATURE_DIM), dtype=np.float32)
    slot_count = scores.shape[1]

    selected = np.zeros((slot_count, 3), dtype=np.float32)
    for index in selected_indices:
        if 0 <= int(index) < count:
            selected = np.maximum(selected, scores[int(index)])
    selected_joint = selected[:, 2]
    selected_coverage = float(selected_joint.mean())
    selected_complete = float((selected_joint >= 0.5).mean())

    rows = []
    for passage_score in scores:
        entity = passage_score[:, 0]
        relation = passage_score[:, 1]
        joint = passage_score[:, 2]
        novelty = np.maximum(joint - selected_joint, 0.0)
        uncovered = 1.0 - selected_joint
        redundancy = np.minimum(joint, selected_joint)
        rows.append((
            float(joint.mean()),
            float(joint.max()),
            float((joint >= 0.5).mean()),
            float(entity.mean()),
            float(relation.mean()),
            float((relation >= 0.5).mean()),
            float(novelty.mean()),
            float((novelty > 0.25).mean()),
            float((joint * uncovered).mean()),
            float(redundancy.mean()),
            selected_coverage,
            selected_complete,
        ))
    return np.asarray(rows, dtype=np.float32)


def query_plan_evidence_features(
        passage_ids: Sequence[object], selected_indices: Sequence[int],
        metadata: Dict[str, Dict], raw_triples: Sequence[Sequence[object]]) -> np.ndarray:
    """Describe how each passage covers currently uncovered QueryPlan slots."""
    return query_plan_features_from_scores(
        query_plan_slot_scores(passage_ids, metadata, raw_triples),
        selected_indices,
    )
