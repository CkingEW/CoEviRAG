"""Candidate, query-state, and set-conditioned evidence features."""
import numpy as np
from ..query_plan_evidence import query_plan_evidence_features, query_plan_features_from_scores

PURE_SELECTOR = np.delete(np.arange(41), (32, 33, 34, 39))
NO_TOP2_SELECTOR = np.delete(PURE_SELECTOR, np.where(np.isin(PURE_SELECTOR, (35, 36, 37))))
PAIR_BASE_COLUMNS = np.asarray((0, 1, 2, 3, 4, 5, 6, 7, 11, 12, 13))


def rankings(components):
    return {
        source: [
            record["passage"] for record in sorted(
                (item for item in components if item[source] is not None),
                key=lambda item: (item[source], item["passage"]),
                reverse=True,
            )
        ]
        for source in ("fact", "paragraph", "similarity")
    }


def candidate_pool(components):
    source_rankings = rankings(components)
    selected = set(source_rankings["fact"][:200])
    selected.update(source_rankings["paragraph"][:200])
    return [record for record in components if record["passage"] in selected]


def candidate_features(components):
    def normalize(key):
        present = [float(item[key]) for item in components if item[key] is not None]
        if not present:
            return {item["passage"]: 0.0 for item in components}, 0.0
        low, high = min(present), max(present)
        values = {
            item["passage"]: (
                (float(item[key]) - low) / (high - low)
                if item[key] is not None and high - low > 1e-12
                else (1.0 if item[key] is not None else 0.0)
            )
            for item in components
        }
        ordered = sorted(values.values(), reverse=True)
        return values, ordered[0] - ordered[1] if len(ordered) > 1 else 1.0

    fact, fact_gap = normalize("fact")
    paragraph, paragraph_gap = normalize("paragraph")
    similarity, similarity_gap = normalize("similarity")
    reciprocal = {
        source: {passage: 1.0 / rank for rank, passage in enumerate(order, 1)}
        for source, order in rankings(components).items()
    }
    rows = []
    for item in components:
        passage = item["passage"]
        fact_present = float(item["fact"] is not None)
        paragraph_present = float(item["paragraph"] is not None)
        count = fact_present + paragraph_present
        rows.append([
            float(item["similarity"]), count, max(0.0, count - 1.0),
            fact_present, paragraph_present,
            fact[passage], paragraph[passage], similarity[passage],
            fact[passage] * fact_gap,
            paragraph[passage] * paragraph_gap,
            similarity[passage] * similarity_gap,
            reciprocal["fact"].get(passage, 0.0),
            reciprocal["paragraph"].get(passage, 0.0),
            reciprocal["similarity"].get(passage, 0.0),
        ])
    return np.asarray(rows, dtype=np.float32)


def current_query_features(components):
    values = {
        source: np.asarray([float(item[source] or 0.0) for item in components])
        for source in ("fact", "paragraph", "similarity")
    }
    available = {
        source: np.asarray([item[source] is not None for item in components])
        for source in values
    }
    features = []
    orders = {}
    for source in values:
        order = np.flatnonzero(available[source])
        order = np.asarray(sorted(
            order.tolist(),
            key=lambda idx: (float(values[source][idx]), idx),
            reverse=True,
        ), dtype=np.int64)
        orders[source] = order
        ranked = values[source][order].tolist()
        padded = ranked[:3] + [0.0] * max(0, 3 - len(ranked))
        features.extend(padded[:3])
        features.append(padded[0] - padded[1])
        features.append(float(np.mean(ranked[:5])) if ranked else 0.0)
        features.append(len(ranked) / max(1.0, float(len(components))))
    features.extend((0.0, 0.0, 0.0))
    top = {source: set(order[:2]) for source, order in orders.items()}
    agreement = np.asarray([float(item["agreement"]) for item in components])
    features.extend((
        len(top["fact"] & top["similarity"]) / 2.0,
        len(top["fact"] & top["paragraph"]) / 2.0,
        len(top["paragraph"] & top["similarity"]) / 2.0,
        float(np.mean(agreement)) if len(agreement) else 0.0,
        0.0,
        min(1.0, len(components) / 200.0),
    ))
    return np.asarray(features, dtype=np.float32)


def multiscale_features(base, include_top20):
    values = {
        "fact": base[:, 5][base[:, 3] > 0.5],
        "paragraph": base[:, 6][base[:, 4] > 0.5],
        "similarity": base[:, 7],
    }
    source_indices = {
        "fact": np.flatnonzero(base[:, 3] > 0.5),
        "paragraph": np.flatnonzero(base[:, 4] > 0.5),
        "similarity": np.arange(len(base)),
    }
    orders = {}
    features = []
    for source in ("fact", "paragraph", "similarity"):
        indices = source_indices[source]
        scores = base[indices, {"fact": 5, "paragraph": 6, "similarity": 7}[source]]
        order = np.asarray(sorted(
            indices.tolist(),
            key=lambda idx: (float(base[idx, {"fact": 5, "paragraph": 6, "similarity": 7}[source]]), idx),
            reverse=True,
        ), dtype=np.int64)
        orders[source] = order
        ranked = base[order, {"fact": 5, "paragraph": 6, "similarity": 7}[source]]
        features.extend((
            float(np.mean(ranked[:10])) if len(ranked) else 0.0,
            float(ranked[4] - ranked[5]) if len(ranked) > 5 else 0.0,
            float(ranked[9] - ranked[10]) if len(ranked) > 10 else 0.0,
        ))
        if include_top20:
            features.append(float(np.mean(ranked[:20])) if len(ranked) else 0.0)
    for left, right in (("fact", "similarity"), ("fact", "paragraph"), ("paragraph", "similarity")):
        for k in (2, 5, 10):
            features.append(len(set(orders[left][:k]) & set(orders[right][:k])) / float(k))
    return np.asarray(features, dtype=np.float32)


def paragraph_evidence_features(components):
    rows = []
    for item in components:
        present = item.get("paragraph_evidence") is not None
        rows.append([
            float(item.get("paragraph_evidence") or 0.0),
            float(item.get("paragraph_policy_probability") or 0.0),
            float(item.get("paragraph_query_path_mean") or 0.0),
            float(item.get("paragraph_seed_score") or 0.0),
            min(1.0, float(item.get("paragraph_hops") or 0.0) / 2.0)
            if present else 0.0,
        ])
    return np.asarray(rows, dtype=np.float32)


def build_candidate_features(components, pool):
    base = candidate_features(pool)
    legacy = np.concatenate((base, np.broadcast_to(
        current_query_features(components), (len(base), 27))), axis=1)
    features = np.concatenate((
        legacy[:, NO_TOP2_SELECTOR], paragraph_evidence_features(pool),
        np.broadcast_to(multiscale_features(base, True), (len(base), 21)),
    ), axis=1).astype(np.float32)
    return base, features


def build_features(exported, query_ids):
    if exported.shape[1] != 46:
        raise ValueError(
            "Expected 46 features (14 candidate + 5 Paragraph + 27 query), "
            f"got {exported.shape[1]}"
        )
    base = exported[:, :14]
    paragraph = exported[:, 14:19]
    legacy = np.concatenate((base, exported[:, 19:]), axis=1)
    multiscale = np.zeros((len(base), 21), dtype=np.float32)
    for query_id in np.unique(query_ids):
        mask = query_ids == query_id
        multiscale[mask] = multiscale_features(base[mask], True)
    return np.concatenate(
        (legacy[:, NO_TOP2_SELECTOR], paragraph, multiscale), axis=1
    ).astype(np.float32)


def semantic_pair_features(passage_ids, anchor_index, metadata):
    anchor = metadata.get(str(passage_ids[anchor_index]), {})
    anchor_entities = anchor.get("entities", set())
    anchor_relations = anchor.get("relations", set())
    anchor_tokens = anchor.get("tokens", set())
    anchor_title = anchor.get("title", "")
    rows = []
    for passage_id in passage_ids:
        candidate = metadata.get(str(passage_id), {})
        entities = candidate.get("entities", set())
        relations = candidate.get("relations", set())
        tokens = candidate.get("tokens", set())
        shared_entities = len(anchor_entities & entities)
        entity_union = len(anchor_entities | entities)
        relation_union = len(anchor_relations | relations)
        token_union = len(anchor_tokens | tokens)
        entity_jaccard = shared_entities / max(1, entity_union)
        token_jaccard = len(anchor_tokens & tokens) / max(1, token_union)
        bridge = float(shared_entities > 0)
        rows.append((
            min(shared_entities, 5) / 5.0,
            entity_jaccard,
            len(anchor_relations & relations) / max(1, relation_union),
            token_jaccard,
            float(bool(anchor_title) and anchor_title in candidate.get("text", "")),
            float(bool(candidate.get("title")) and candidate["title"] in anchor.get("text", "")),
            bridge,
            bridge * (1.0 - token_jaccard),
        ))
    return np.asarray(rows, dtype=np.float32)


def conditional_features(
        base, scores, anchor_index, passage_ids=None, openie_metadata=None,
        raw_query_triples=None, selected_indices=None, query_plan_scores=None):
    candidate = base[:, PAIR_BASE_COLUMNS]
    anchor = np.broadcast_to(candidate[anchor_index], candidate.shape)
    candidate_scores = scores.reshape(-1, 1)
    anchor_scores = np.full_like(candidate_scores, scores[anchor_index])
    pair = np.column_stack((
        base[:, 3] * base[anchor_index, 3],
        base[:, 4] * base[anchor_index, 4],
        base[:, 3] * base[anchor_index, 4] + base[:, 4] * base[anchor_index, 3],
        np.abs(base[:, 5] - base[anchor_index, 5]),
        np.abs(base[:, 6] - base[anchor_index, 6]),
        np.abs(base[:, 7] - base[anchor_index, 7]),
        np.abs(scores - scores[anchor_index]),
    ))
    features = [candidate, anchor, candidate_scores, anchor_scores, pair]
    if openie_metadata is not None:
        if passage_ids is None:
            raise ValueError(
                "Semantic pair features require passage_ids in the candidate cache"
            )
        features.append(semantic_pair_features(
            passage_ids, anchor_index, openie_metadata
        ))
        if raw_query_triples is not None:
            selected = selected_indices if selected_indices is not None else [anchor_index]
            features.append(
                query_plan_features_from_scores(query_plan_scores, selected)
                if query_plan_scores is not None else
                query_plan_evidence_features(
                    passage_ids, selected, openie_metadata, raw_query_triples
                )
            )
    return np.concatenate(features, axis=1).astype(np.float32)


def normalized_unary_prior(scores, candidate_mask):
    values = scores[candidate_mask].astype(np.float32)
    scale = float(values.std())
    if scale <= 1e-6:
        return np.zeros_like(values)
    return (values - float(values.mean())) / scale
