# Modified for CoEviRAG: release scope, portable paths, or fixed-protocol guards.
import argparse
import gzip
import json
import random
from pathlib import Path

import numpy as np

from evidence_features import (
    NO_TOP2_SELECTOR,
    candidate_features,
    candidate_pool,
    current_query_features,
    gold_docs,
    multiscale_features,
    train_and_evaluate,
)


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


def main():
    parser = argparse.ArgumentParser(
        description="Train a ranker with separated Paragraph search/evidence signals."
    )
    parser.add_argument("--train_cache", required=True)
    parser.add_argument("--val_components", required=True)
    parser.add_argument("--val_data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--dataset", default="2wikimultihopqa")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--ranker_objective",
        choices=("listwise", "top_heavy", "multiscale_boundary"),
        default="listwise",
    )
    parser.add_argument("--pairwise_weight", type=float, default=0.5)
    parser.add_argument("--pairwise_margin", type=float, default=0.1)
    parser.add_argument("--hard_negative_count", type=int, default=20)
    parser.add_argument("--ranker_lr", type=float, default=0.005)
    parser.add_argument("--init_checkpoint", default="")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    cache = np.load(args.train_cache)
    exported = cache["features"].astype(np.float32)
    if exported.shape[1] != 46:
        raise ValueError(
            "Expected 46 exported features (14 candidate + 5 Paragraph "
            f"evidence + 27 query), got {exported.shape[1]}"
        )
    labels = cache["labels"].astype(np.float32)
    query_ids = cache["query_ids"].astype(np.int64)

    train_base = exported[:, :14]
    train_paragraph = exported[:, 14:19]
    train_legacy_full = np.concatenate((train_base, exported[:, 19:]), axis=1)
    train_multiscale = np.zeros((len(train_base), 21), dtype=np.float32)
    for query_id in np.unique(query_ids):
        mask = query_ids == query_id
        train_multiscale[mask] = multiscale_features(train_base[mask], True)
    train_x = np.concatenate((
        train_legacy_full[:, NO_TOP2_SELECTOR],
        train_paragraph,
        train_multiscale,
    ), axis=1)

    with gzip.open(args.val_components, "rt", encoding="utf-8") as file:
        component_sets = json.load(file)
    pools = [candidate_pool(items) for items in component_sets]
    val_x = []
    for components, pool in zip(component_sets, pools):
        base = candidate_features(pool)
        legacy_full = np.concatenate((
            base,
            np.broadcast_to(
                current_query_features(components),
                (len(base), 27),
            ),
        ), axis=1)
        val_x.append(np.concatenate((
            legacy_full[:, NO_TOP2_SELECTOR],
            paragraph_evidence_features(pool),
            np.broadcast_to(
                multiscale_features(base, True),
                (len(base), 21),
            ),
        ), axis=1))

    samples = json.load(Path(args.val_data).open(encoding="utf-8"))[:len(pools)]
    targets = gold_docs(samples, args.dataset)
    result = train_and_evaluate(
        "paragraph_evidence_multiscale60",
        train_x,
        labels,
        query_ids,
        val_x,
        pools,
        targets,
        args,
        NO_TOP2_SELECTOR,
        "paragraph_evidence_v1_multiscale_k2_5_10_20_v1",
    )
    payload = {"seed": args.seed, "epochs": args.epochs, "result": result}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
