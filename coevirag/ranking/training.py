"""Per-dataset unary and initial-selector training and validation."""
import json
import numpy as np
import torch
from . import RECALL_KS
from .features import normalized_unary_prior
from .models import make_model, unary_scores
from .selection import select_ranking, summarize
from ..release_protocol import BENCHMARK_DATASETS as DATASETS


def checkpoint(model, mean, std, epoch, selector=False, datasets=DATASETS):
    return {
        "model_state_dict": {key: value.detach().cpu().clone()
                             for key, value in model.state_dict().items()},
        "feature_mean": torch.from_numpy(mean), "feature_std": torch.from_numpy(std),
        "input_dim": len(mean), "hidden_dim": 16, "epoch": epoch,
        "feature_schema": ("anchor_bridge_queryplan51_residual_v3" if selector else
                           "paragraph_evidence_v1_multiscale_k2_5_10_20_v1"),
        "unary_mode": "dpr", "selector_mode": "residual",
        "uses_query_plan_features": selector, "bridge_candidate_limit": 0,
        "training_datasets": list(datasets), "training_candidate_pool": "dual_union_dpr200",
        "selection_metric": "macro_dataset_mean_recall_2_5_10",
    }


def padded_listwise_loss(model, rows, device):
    width = max(len(row["y"]) for row in rows)
    dim = rows[0]["x"].shape[1]
    x = np.zeros((len(rows), width, dim), dtype=np.float32)
    target = np.zeros((len(rows), width), dtype=bool)
    mask = np.zeros_like(target)
    prior = np.zeros_like(target, dtype=np.float32)
    for index, row in enumerate(rows):
        size = len(row["y"])
        x[index, :size] = row["x"]
        target[index, :size] = row["y"]
        mask[index, :size] = True
        prior[index, :size] = row.get("prior", 0.0)
    mask = torch.from_numpy(mask).to(device)
    y = torch.from_numpy(target).to(device)
    scores = model(torch.from_numpy(x).to(device)).squeeze(-1)
    scores = (scores + torch.from_numpy(prior).to(device)).masked_fill(~mask, -torch.inf)
    log_probs = torch.log_softmax(scores, dim=1)
    losses = -log_probs.masked_fill(~y, 0.0).sum(1) / y.sum(1).clamp_min(1)
    weights = torch.tensor([row["weight"] for row in rows], device=device)
    return (losses * weights).mean()


def balanced_rows(bundles, selector=False):
    rows = []
    for dataset, bundle in bundles.items():
        data = bundle["train"]
        for group in bundle["train_groups"]:
            mask = np.ones(len(group["labels"]), dtype=bool)
            if selector:
                mask[group["anchor"]] = False
            y = group["labels"][mask]
            if not y.any():
                continue
            x = (group["pair_features"][mask] if selector else
                 data["features"][group["indices"]][mask])
            row = {"dataset": dataset, "x": x, "y": y}
            if selector:
                row["prior"] = normalized_unary_prior(group["scores"], mask)
            rows.append(row)
    counts = {dataset: sum(row["dataset"] == dataset for row in rows) for dataset in bundles}
    if not all(counts.values()):
        raise ValueError("A dataset has no eligible training queries")
    for row in rows:
        row["weight"] = len(rows) / (len(bundles) * counts[row["dataset"]])
    # Normalization uses training features only; no validation/test statistics.
    x = np.concatenate([row["x"] for row in rows])
    mean = x.mean(0)
    std = x.std(0)
    std = np.where(std > 1e-6, std, 1.0)
    for row in rows:
        row["x"] = ((row["x"] - mean) / std).astype(np.float32)
    return rows, mean.astype(np.float32), std.astype(np.float32), counts


def evaluate_backend(bundles, tail_ckpt, selector_ckpt=None, anchor_only=False):
    if anchor_only and selector_ckpt is not None:
        raise ValueError("Anchor-only control must not use a conditional selector")
    tail = make_model(60, "cpu")
    tail.load_state_dict(tail_ckpt["model_state_dict"])
    tail.eval()
    selector = None
    if selector_ckpt is not None:
        selector = make_model(int(selector_ckpt["input_dim"]), "cpu")
        selector.load_state_dict(selector_ckpt["model_state_dict"])
        selector.eval()
    results = {}
    for dataset, bundle in bundles.items():
        values, rankings = [], []
        data = bundle["val"]
        all_tail = unary_scores(data["features"], tail, tail_ckpt, "cpu")
        for group in bundle["val_groups"]:
            indices = group["indices"]
            qid = group["query_id"]
            if selector is None:
                order = np.argsort(-all_tail[indices], kind="stable")
                if anchor_only:
                    anchor = int(group["anchor"])
                    order = np.concatenate(([anchor], order[order != anchor]))
            else:
                order = select_ranking(
                    data["base"][indices], data["passage_ids"][indices],
                    group["scores"], selector, selector_ckpt, "setmax",
                    min(4, bundle["budgets"][qid] + 1), bundle["metadata"], "cpu",
                    tail_scores=all_tail[indices], raw_query_triples=bundle["plans"][qid],
                    query_plan_scores=group["plan_scores"],
                )
            labels = group["labels"][order]
            denominator = bundle["denominators"][qid]
            values.append([float(labels[:k].sum()) / denominator for k in RECALL_KS])
            rankings.append({"query_id": qid, "passage_ids":
                             data["passage_ids"][indices][order[:200]].tolist()})
        results[dataset] = summarize(values)
        results[dataset]["rankings"] = rankings
    macro = float(np.mean([value["mean_recall_2_5_10"] for value in results.values()]))
    return macro, results


def train_stage(bundles, root, epochs, batch_size, seed, selector=False, tail_ckpt=None):
    rows, mean, std, counts = balanced_rows(bundles, selector)
    torch.manual_seed(seed)
    model = make_model(len(mean), "cuda", residual=selector)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003 if selector else 0.005,
                                 weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    best_score, best, history = -np.inf, None, []
    stage = "selector" if selector else "tail"
    print(f"TRAIN {stage} eligible_queries={counts}", flush=True)
    for epoch in range(epochs + 1):
        if epoch:
            model.train()
            order = rng.permutation(len(rows))
            losses = []
            for start in range(0, len(rows), batch_size):
                batch = [rows[int(index)] for index in order[start:start + batch_size]]
                optimizer.zero_grad(set_to_none=True)
                loss = padded_listwise_loss(model, batch, "cuda")
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
        current = checkpoint(model, mean, std, epoch, selector, datasets=tuple(bundles))
        macro, results = evaluate_backend(
            bundles, tail_ckpt if selector else current, current if selector else None)
        compact = {dataset: value["metrics"] for dataset, value in results.items()}
        history.append(dict(epoch=epoch, macro=macro, validation=compact,
                            loss=float(np.mean(losses)) if epoch else None))
        if macro > best_score:
            best_score, best = macro, current
            best["validation_macro"] = macro
            prefix = "independent" if len(bundles) == 1 else "shared"
            torch.save(best, root / f"{prefix}_{stage}_seed{seed}.pth")
        (root / f"{stage}_history.json").write_text(json.dumps(history, indent=2))
        heads = {dataset: [round(100 * metric[f"Recall@{k}"], 2) for k in (2, 5, 10)]
                 for dataset, metric in compact.items()}
        print(f"EPOCH {stage} {epoch}/{epochs} macro={macro:.6f} best={best_score:.6f} {heads}",
              flush=True)
    return best
