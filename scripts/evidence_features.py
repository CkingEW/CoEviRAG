# Modified for CoEviRAG: retain shared feature/training helpers only.
import argparse

import gzip

import json

import random

from pathlib import Path

import numpy as np

import torch

K_LIST = list(range(1, 201))

PURE_SELECTOR = np.delete(np.arange(41), (32, 33, 34, 39))

NO_TOP2_SELECTOR = np.delete(PURE_SELECTOR, np.where(np.isin(PURE_SELECTOR, (35, 36, 37))))

def gold_docs(samples, dataset):
    output = []
    for sample in samples:
        if "supporting_facts" in sample:
            titles = {item[0] for item in sample["supporting_facts"]}
            separator = "" if dataset.startswith("hotpotqa") else " "
            docs = [
                item[0] + "\n" + separator.join(item[1])
                for item in sample["context"] if item[0] in titles
            ]
        elif "contexts" in sample:
            docs = [
                item["title"] + "\n" + item["text"]
                for item in sample["contexts"] if item["is_supporting"]
            ]
        else:
            docs = [
                item["title"] + "\n" + item.get("text", item.get("paragraph_text", ""))
                for item in sample["paragraphs"] if item.get("is_supporting", True)
            ]
        output.append(list(set(docs)))
    return output

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

def train_and_evaluate(
        name, train_x, labels, query_ids, val_x, pools, gold, args,
        feature_selector, feature_schema):
    mean = train_x.mean(axis=0)
    std = np.where(train_x.std(axis=0) > 1e-6, train_x.std(axis=0), 1.0)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    x = torch.from_numpy((train_x - mean) / std).float().to(device)
    y = torch.from_numpy(labels.astype(bool)).to(device)
    groups = [torch.from_numpy(np.flatnonzero(query_ids == qid)).long().to(device) for qid in np.unique(query_ids)]
    torch.manual_seed(args.seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(train_x.shape[1], 16), torch.nn.ReLU(), torch.nn.Linear(16, 1)
    ).to(device)
    init_checkpoint = str(getattr(args, "init_checkpoint", "") or "")
    if init_checkpoint:
        initial = torch.load(init_checkpoint, map_location="cpu", weights_only=True)
        initial_state = initial["model_state_dict"]
        initial_dim = int(initial["input_dim"])
        if initial_dim == train_x.shape[1]:
            model.load_state_dict(initial_state)
        elif initial_dim == 55 and train_x.shape[1] == 60:
            state = model.state_dict()
            old_weight = initial_state["0.weight"]
            state["0.weight"][:, :34] = old_weight[:, :34]
            state["0.weight"][:, 34:39] = 0.0
            state["0.weight"][:, 39:] = old_weight[:, 34:]
            state["0.bias"] = initial_state["0.bias"]
            state["2.weight"] = initial_state["2.weight"]
            state["2.bias"] = initial_state["2.bias"]
            model.load_state_dict(state)
        else:
            raise ValueError(
                f"Unsupported warm start dimensions: {initial_dim} -> "
                f"{train_x.shape[1]}"
            )
    learning_rate = float(getattr(args, "ranker_lr", 0.005))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    objective = getattr(args, "ranker_objective", "listwise")
    pairwise_weight = float(getattr(args, "pairwise_weight", 0.0))
    pairwise_margin = float(getattr(args, "pairwise_margin", 0.1))
    hard_negative_count = max(1, int(getattr(args, "hard_negative_count", 20)))
    for _ in range(args.epochs):
        for indices in groups:
            positives = y[indices]
            if not bool(positives.any()):
                continue
            optimizer.zero_grad()
            scores = model(x[indices]).view(-1)
            loss = -torch.log_softmax(scores, dim=0)[positives].mean()
            if objective == "top_heavy" and bool((~positives).any()):
                positive_scores = scores[positives]
                negative_scores = scores[~positives]
                hard_negatives = torch.topk(
                    negative_scores,
                    k=min(hard_negative_count, negative_scores.numel()),
                ).values
                pairwise_loss = torch.nn.functional.softplus(
                    pairwise_margin
                    + hard_negatives.view(1, -1)
                    - positive_scores.view(-1, 1)
                ).mean()
                loss = loss + pairwise_weight * pairwise_loss
            elif objective == "multiscale_boundary" and bool((~positives).any()):
                positive_scores = torch.sort(
                    scores[positives], descending=True
                ).values
                negative_scores = torch.sort(
                    scores[~positives], descending=True
                ).values
                boundary_losses = []
                for cutoff in (2, 5, 10):
                    required_positives = min(
                        cutoff, int(positive_scores.numel())
                    )
                    negative_index = cutoff - required_positives
                    if (
                            required_positives <= 0
                            or negative_index >= negative_scores.numel()):
                        continue
                    positive_boundary = positive_scores[required_positives - 1]
                    negative_boundary = negative_scores[negative_index]
                    boundary_losses.append(torch.nn.functional.softplus(
                        pairwise_margin
                        + negative_boundary
                        - positive_boundary
                    ))
                if boundary_losses:
                    loss = loss + pairwise_weight * torch.stack(
                        boundary_losses
                    ).mean()
            loss.backward()
            optimizer.step()
    ranked_docs = []
    with torch.no_grad():
        for features, pool in zip(val_x, pools):
            z = torch.from_numpy((features - mean) / std).float().to(device)
            scores = model(z).view(-1).cpu().numpy()
            order = np.argsort(-scores, kind="stable")
            ranked_docs.append([pool[idx]["content"] for idx in order])
    metrics = {}
    for k in K_LIST:
        recalls = [len(set(docs[:k]) & set(target)) / len(set(target)) for docs, target in zip(ranked_docs, gold)]
        metrics[f"Recall@{k}"] = round(float(np.mean(recalls)), 4)
    composite = round(float(np.mean([metrics[f"Recall@{k}"] for k in (2, 5, 10)])), 6)
    checkpoint_path = None
    if args.checkpoint_dir:
        checkpoint_path = Path(args.checkpoint_dir) / f"{name}_seed{args.seed}.pth"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "feature_mean": torch.from_numpy(mean.astype(np.float32)),
            "feature_std": torch.from_numpy(std.astype(np.float32)),
            "model_state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "input_dim": int(train_x.shape[1]),
            "hidden_dim": 16,
            "objective": objective,
            "pairwise_weight": pairwise_weight,
            "pairwise_margin": pairwise_margin,
            "hard_negative_count": hard_negative_count,
            "learning_rate": learning_rate,
            "warm_start_checkpoint": init_checkpoint,
            "epochs": args.epochs,
            "fact_candidate_budget": 200,
            "paragraph_candidate_budget": 200,
            "feature_selector": torch.from_numpy(feature_selector.astype(np.int64)),
            "feature_schema": feature_schema,
        }, checkpoint_path)
    return {
        "name": name,
        "input_dim": train_x.shape[1],
        "metrics": metrics,
        "composite": composite,
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
    }
