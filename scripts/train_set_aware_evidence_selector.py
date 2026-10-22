# Modified for CoEviRAG: release scope, portable paths, or fixed-protocol guards.
import argparse
import gzip
import json
import random
import re
from pathlib import Path
import sys

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from cache_features import build_features
from evidence_features import (
    candidate_features,
    candidate_pool,
    current_query_features,
    gold_docs,
    multiscale_features,
    NO_TOP2_SELECTOR,
)
from train_paragraph_evidence_ranker import paragraph_evidence_features
from coevirag.query_plan_evidence import (
    valid_openie_triples,
    query_plan_evidence_features,
    query_plan_features_from_scores,
    query_plan_slot_scores,
)


RECALL_KS = (1, 2, 5, 10, 20, 30, 50, 100, 150, 200)
PAIR_BASE_COLUMNS = np.asarray((0, 1, 2, 3, 4, 5, 6, 7, 11, 12, 13))
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "in", "is", "it", "of", "on", "or", "that", "the", "this", "to",
    "was", "were", "with",
}


def load_unary_ranker(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = torch.nn.Sequential(
        torch.nn.Linear(int(checkpoint["input_dim"]), int(checkpoint["hidden_dim"])),
        torch.nn.ReLU(),
        torch.nn.Linear(int(checkpoint["hidden_dim"]), 1),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def unary_scores(features, model, checkpoint, device):
    mean = checkpoint["feature_mean"].cpu().numpy()
    std = checkpoint["feature_std"].cpu().numpy()
    with torch.no_grad():
        return model(torch.from_numpy((features - mean) / std).float().to(device)).view(-1).cpu().numpy()


def candidate_unary_scores(data, indices, mode, model, checkpoint, device):
    if mode == "dpr":
        return data["base"][indices, 0].astype(np.float32)
    return unary_scores(data["features"][indices], model, checkpoint, device)


def normalized_terms(text):
    return {token for token in TOKEN_PATTERN.findall(str(text).lower()) if token not in STOPWORDS}


def load_openie_metadata(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    documents = payload.get("docs", []) if isinstance(payload, dict) else payload
    metadata = {}
    for document in documents:
        passage = str(document.get("passage", ""))
        title = passage.split("\n", 1)[0].strip().lower()
        entities = {
            " ".join(TOKEN_PATTERN.findall(str(entity).lower()))
            for entity in document.get("extracted_entities", [])
        }
        entities.discard("")
        relations = {
            " ".join(TOKEN_PATTERN.findall(str(triple[1]).lower()))
            for triple in document.get("extracted_triples", []) if len(triple) >= 3
        }
        relations.discard("")
        metadata[str(document["idx"])] = {
            "title": title,
            "content": passage,
            "text": passage.lower(),
            "entities": entities,
            "relations": relations,
            "tokens": normalized_terms(passage),
            "triples": valid_openie_triples(document.get("extracted_triples", [])),
        }
    return metadata


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


def load_query_plans(split_path, query_cache_path, query_ids):
    samples = json.loads(Path(split_path).read_text(encoding="utf-8"))
    cache = json.loads(Path(query_cache_path).read_text(encoding="utf-8"))
    return {
        int(query_id): cache.get(samples[int(query_id)]["question"], {}).get(
            "raw_triples", []
        )
        for query_id in np.unique(query_ids)
    }


def load_npz_cache(path):
    cache = np.load(path)
    exported = cache["features"].astype(np.float32)
    query_ids = cache["query_ids"].astype(np.int64)
    return {
        "base": exported[:, :14],
        "features": build_features(exported, query_ids),
        "labels": cache["labels"].astype(bool),
        "query_ids": query_ids,
        "passage_ids": cache["passage_ids"].astype(str)
        if "passage_ids" in cache.files else None,
    }


def stream_component_sets(path, expected_count):
    import ijson
    count = 0
    with gzip.open(path, "rb") as file:
        for components in ijson.items(file, "item", use_float=True):
            if count >= expected_count:
                raise ValueError("Component cache has more queries than split")
            count += 1
            yield components
            if count % 100 == 0:
                print(f"CACHE_READ {count}/{expected_count}", flush=True)
    if count != expected_count:
        raise ValueError("Component cache has fewer queries than split")


def load_component_cache(
        path, split_path, dataset, dpr_safety_cache="", dpr_safety_top_k=0,
        repair_top_k=0, repair_reference_cache="", stream=False):
    if repair_top_k > 0 and not repair_reference_cache:
        raise ValueError("repair_top_k requires a paired Repair-off reference cache")
    samples = json.loads(Path(split_path).read_text(encoding="utf-8"))
    if stream:
        component_count = len(samples)
        component_sets = stream_component_sets(path, component_count)
    else:
        with gzip.open(path, "rt", encoding="utf-8") as file:
            component_sets = json.load(file)
        component_count = len(component_sets)
        samples = samples[:component_count]
    reference_sets = None
    if repair_reference_cache:
        with gzip.open(repair_reference_cache, "rt", encoding="utf-8") as file:
            reference_sets = json.load(file)
        if len(reference_sets) != component_count:
            raise ValueError("Repair reference cache must contain the same queries")
    safety = None
    if dpr_safety_cache:
        # NpzFile indexing decompresses on every access; materialize once per run.
        with np.load(dpr_safety_cache) as archive:
            safety = {key: archive[key] for key in archive.files}
    safety_contents = None
    if safety is not None:
        if safety["passage_ids"].shape != safety["scores"].shape:
            raise ValueError("DPR safety IDs and scores have incompatible shapes")
        if "contents_json" in safety:
            safety_contents = json.loads(str(safety["contents_json"].item()))
    if safety is not None and len(safety["passage_ids"]) < component_count:
        raise ValueError("DPR safety cache has fewer queries than component cache")
    if safety is not None and "questions" in safety:
        cached_questions = safety["questions"]
        if len(cached_questions) < len(samples) or any(
                " ".join(str(question).split()) != " ".join(sample["question"].split())
                for question, sample in zip(cached_questions, samples)):
            raise ValueError("DPR safety questions do not match split order")
    targets = gold_docs(samples, dataset)
    base_rows = []
    feature_rows = []
    labels = []
    query_ids = []
    passage_ids = []
    for query_id, (components, target) in enumerate(zip(component_sets, targets)):
        components = list(components)
        pool = candidate_pool(components)
        if repair_top_k > 0:
            selected = {str(item["passage"]) for item in pool}
            reference_ids = (
                {str(item["passage"]) for item in reference_sets[query_id]}
                if reference_sets is not None else None
            )
            repair_candidates = sorted(
                (item for item in components if (
                    str(item["passage"]) not in reference_ids
                    if reference_ids is not None else float(item.get("repair") or 0) > 0
                )),
                key=lambda item: (float(item["repair"]), str(item["passage"])),
                reverse=True,
            )[:repair_top_k]
            for item in repair_candidates:
                passage = str(item["passage"])
                if passage not in selected:
                    pool.append(item)
                    selected.add(passage)
        if safety is not None and dpr_safety_top_k > 0:
            by_passage = {str(item["passage"]): item for item in components}
            selected = {str(item["passage"]) for item in pool}
            width = min(dpr_safety_top_k, safety["passage_ids"].shape[1])
            for rank in range(width):
                passage = str(safety["passage_ids"][query_id, rank])
                record = by_passage.get(passage)
                if record is None:
                    record = {
                        "passage": passage,
                        "fact": None,
                        "paragraph": None,
                        "similarity": float(safety["scores"][query_id, rank]),
                        "ppr": 0.0,
                        "repair": 0.0,
                        "agreement": 0.0,
                        "content": (safety_contents[passage] if safety_contents is not None
                                    else str(safety["contents"][query_id, rank])),
                    }
                    components.append(record)
                    by_passage[passage] = record
                if passage not in selected:
                    pool.append(record)
                    selected.add(passage)
        base = candidate_features(pool)
        legacy = np.concatenate((
            base,
            np.broadcast_to(current_query_features(components), (len(base), 27)),
        ), axis=1)
        features = np.concatenate((
            legacy[:, NO_TOP2_SELECTOR],
            paragraph_evidence_features(pool),
            np.broadcast_to(multiscale_features(base, True), (len(base), 21)),
        ), axis=1).astype(np.float32)
        target_set = set(target)
        contents = [item.get("content") or item.get("info", {}).get("content") for item in pool]
        base_rows.append(base)
        feature_rows.append(features)
        labels.extend(content in target_set for content in contents)
        query_ids.extend([query_id] * len(pool))
        passage_ids.extend(str(item["passage"]) for item in pool)
    return {
        "base": np.concatenate(base_rows),
        "features": np.concatenate(feature_rows),
        "labels": np.asarray(labels, dtype=bool),
        "query_ids": np.asarray(query_ids, dtype=np.int64),
        "passage_ids": np.asarray(passage_ids, dtype=str),
    }


def gold_denominators(split_path, query_ids):
    samples = json.loads(Path(split_path).read_text(encoding="utf-8"))
    denominators = {}
    for query_id in np.unique(query_ids):
        sample = samples[int(query_id)]
        if "contexts" in sample:
            count = sum(bool(item.get("is_supporting")) for item in sample["contexts"])
        elif "supporting_facts" in sample:
            count = len({item[0] for item in sample["supporting_facts"]})
        else:
            count = sum(bool(item.get("is_supporting", True)) for item in sample["paragraphs"])
        denominators[int(query_id)] = max(1, count)
    return denominators


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


def normalized_unary_prior(scores, candidate_mask):
    values = scores[candidate_mask].astype(np.float32)
    scale = float(values.std())
    if scale <= 1e-6:
        return np.zeros_like(values)
    return (values - float(values.mean())) / scale


def train_selector(
        groups, epochs, seed, hidden_dim, learning_rate, first_hit_weight,
        complete_weight, selector_mode, device):
    training_rows = []
    for group in groups:
        labels = group["labels"]
        anchor = group["anchor"]
        target = labels.copy()
        target[anchor] = False
        if labels[anchor]:
            mode_weight = complete_weight
        else:
            target = labels.copy()
            target[anchor] = False
            mode_weight = first_hit_weight
        if target.any():
            training_rows.append((group, target, mode_weight))

    normalization_features = []
    for group, _, _ in training_rows:
        candidate_mask = (
            (np.arange(len(group["labels"])) != group["anchor"])
            & group["eligible"]
        )
        normalization_features.append(group["pair_features"][candidate_mask])
    all_features = np.concatenate(normalization_features)
    mean = all_features.mean(axis=0)
    std = np.where(all_features.std(axis=0) > 1e-6, all_features.std(axis=0), 1.0)
    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(all_features.shape[1], hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, 1),
    ).to(device)
    if selector_mode == "residual":
        torch.nn.init.zeros_(model[2].weight)
        torch.nn.init.zeros_(model[2].bias)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    rng = random.Random(seed)
    for _ in range(epochs):
        rng.shuffle(training_rows)
        for group, target, weight in training_rows:
            candidate_mask = (
                (np.arange(len(target)) != group["anchor"])
                & group["eligible"]
            )
            x = torch.from_numpy((group["pair_features"][candidate_mask] - mean) / std).float().to(device)
            y = torch.from_numpy(target[candidate_mask]).bool().to(device)
            optimizer.zero_grad()
            scores = model(x).view(-1)
            if selector_mode == "residual":
                prior = torch.from_numpy(normalized_unary_prior(
                    group["scores"], candidate_mask
                )).float().to(device)
                scores = scores + prior
            loss = -torch.log_softmax(scores, dim=0)[y].mean() * weight
            loss.backward()
            optimizer.step()
    return model, mean.astype(np.float32), std.astype(np.float32), len(training_rows)


def evaluate(groups, model, mean, std, denominators, selector_mode, device):
    baseline_values = []
    selector_values = []
    states = {
        name: {
            "queries": 0,
            "anchor_hit": 0,
            "zero_hit_at2": 0,
            "one_hit_at2": 0,
            "complete_at2": 0,
        }
        for name in ("baseline", "set_aware")
    }
    with torch.no_grad():
        for group in groups:
            labels = group["labels"]
            baseline = group["order"]
            anchor = group["anchor"]
            candidate_indices = np.flatnonzero(
                (np.arange(len(labels)) != anchor) & group["eligible"]
            )
            x = torch.from_numpy((group["pair_features"][candidate_indices] - mean) / std).float().to(device)
            bridge_scores = model(x).view(-1).cpu().numpy()
            if selector_mode == "residual":
                candidate_mask = (
                    (np.arange(len(labels)) != anchor) & group["eligible"]
                )
                bridge_scores += normalized_unary_prior(
                    group["scores"], candidate_mask
                )
            bridge = int(candidate_indices[int(np.argmax(bridge_scores))])
            selected = [anchor, bridge]
            selected.extend(int(idx) for idx in baseline if idx not in (anchor, bridge))
            denominator = denominators[group["query_id"]]
            baseline_values.append([float(labels[baseline[:k]].sum()) / denominator for k in RECALL_KS])
            selector_values.append([float(labels[selected[:k]].sum()) / denominator for k in RECALL_KS])
            for name, ranking in (("baseline", baseline), ("set_aware", selected)):
                hits = int(labels[ranking[:2]].sum())
                states[name]["queries"] += 1
                states[name]["anchor_hit"] += int(labels[anchor])
                states[name]["zero_hit_at2"] += int(hits == 0)
                states[name]["one_hit_at2"] += int(hits == 1)
                states[name]["complete_at2"] += int(hits >= denominator)
    baseline_values = np.asarray(baseline_values, dtype=np.float32)
    selector_values = np.asarray(selector_values, dtype=np.float32)

    def summarize(values):
        metrics = {f"Recall@{k}": float(values[:, index].mean()) for index, k in enumerate(RECALL_KS)}
        return {
            "metrics": metrics,
            "mean_recall_2_5_10": float(values[:, (1, 2, 3)].mean()),
            "per_query_recall": values.tolist(),
        }

    return {"baseline": summarize(baseline_values), "set_aware": summarize(selector_values), "states": states}


def main():
    parser = argparse.ArgumentParser(description="Train a conditional anchor-bridge evidence selector.")
    parser.add_argument("--train_cache", required=True)
    parser.add_argument("--val_components", required=True)
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--val_data", required=True)
    parser.add_argument("--test_cache", default="")
    parser.add_argument("--test_data", default="")
    parser.add_argument("--unary_checkpoint", default="")
    parser.add_argument(
        "--unary_mode", choices=("learned", "dpr"), default="learned"
    )
    parser.add_argument(
        "--openie_results", default="",
        help="Optional OpenIE JSON used for entity/relation pair features",
    )
    parser.add_argument(
        "--query_cache", default="",
        help="Optional QueryPlan cache; enables query-slot coverage features",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", default="hotpotqa")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden_dim", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=0.003)
    parser.add_argument("--first_hit_weight", type=float, default=1.0)
    parser.add_argument("--complete_weight", type=float, default=1.0)
    parser.add_argument(
        "--selector_mode", choices=("direct", "residual"), default="residual"
    )
    parser.add_argument(
        "--bridge_candidate_limit", type=int, default=0,
        help="Restrict conditional second-passage selection to unary Top-k",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    if args.unary_mode == "learned":
        if not args.unary_checkpoint:
            raise ValueError("--unary_checkpoint is required for learned unary mode")
        unary_model, unary_checkpoint = load_unary_ranker(
            args.unary_checkpoint, device
        )
    else:
        unary_model, unary_checkpoint = None, None
    openie_metadata = (
        load_openie_metadata(args.openie_results) if args.openie_results else None
    )
    train = load_npz_cache(args.train_cache)
    val = load_component_cache(args.val_components, args.val_data, args.dataset)
    if args.query_cache and not openie_metadata:
        raise ValueError("--query_cache requires --openie_results")
    train_query_plans = (
        load_query_plans(args.train_data, args.query_cache, train["query_ids"])
        if args.query_cache else None
    )
    val_query_plans = (
        load_query_plans(args.val_data, args.query_cache, val["query_ids"])
        if args.query_cache else None
    )
    train_groups = prepare_groups(
        train, unary_model, unary_checkpoint, device, openie_metadata,
        args.bridge_candidate_limit, args.unary_mode, train_query_plans,
    )
    val_groups = prepare_groups(
        val, unary_model, unary_checkpoint, device, openie_metadata,
        args.bridge_candidate_limit, args.unary_mode, val_query_plans,
    )
    model, mean, std, trained_queries = train_selector(
        train_groups, args.epochs, args.seed, args.hidden_dim, args.learning_rate,
        args.first_hit_weight, args.complete_weight, args.selector_mode, device,
    )
    payload = {
        "config": vars(args),
        "feature_dim": int(mean.size),
        "trained_queries": trained_queries,
        "validation": evaluate(
            val_groups, model, mean, std,
            gold_denominators(args.val_data, val["query_ids"]),
            args.selector_mode, device,
        ),
    }
    if args.test_cache and args.test_data:
        test = load_npz_cache(args.test_cache)
        test_groups = prepare_groups(
            test, unary_model, unary_checkpoint, device, openie_metadata,
            args.bridge_candidate_limit, args.unary_mode,
            load_query_plans(args.test_data, args.query_cache, test["query_ids"])
            if args.query_cache else None,
        )
        payload["test"] = evaluate(
            test_groups, model, mean, std,
            gold_denominators(args.test_data, test["query_ids"]),
            args.selector_mode, device,
        )

    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "feature_mean": torch.from_numpy(mean),
        "feature_std": torch.from_numpy(std),
        "input_dim": int(mean.size),
        "hidden_dim": args.hidden_dim,
        "unary_checkpoint": args.unary_checkpoint,
        "unary_mode": args.unary_mode,
        "feature_schema": (
            f"anchor_bridge_{'queryplan51' if args.query_cache else ('semantic39' if openie_metadata else 'compact31')}_"
            f"{args.selector_mode}_v3"
        ),
        "uses_query_plan_features": bool(args.query_cache),
        "first_hit_weight": args.first_hit_weight,
        "complete_weight": args.complete_weight,
        "selector_mode": args.selector_mode,
        "bridge_candidate_limit": args.bridge_candidate_limit,
    }, checkpoint_path)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    concise = {
        split: {
            key: value["metrics"] | {"mean": value["mean_recall_2_5_10"]}
            for key, value in payload[split].items() if key in ("baseline", "set_aware")
        } | {"states": payload[split]["states"]}
        for split in ("validation", "test") if split in payload
    }
    print(json.dumps(concise, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
