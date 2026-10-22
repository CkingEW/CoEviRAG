# Modified for CoEviRAG: release scope, portable paths, or fixed-protocol guards.
"""Train 51D/53D selectors on inference-aligned selected-set states."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from evaluate_iterative_evidence_selector import query_budgets
from coevirag.query_plan_evidence import (
    append_selection_progress, selection_progress_values,
    query_plan_slot_scores,
    query_plan_state_features,
    mask_query_plan_features,
)
from train_mixed_evidence_backend import (
    DATASETS, assert_split_separation, checkpoint, evaluate_backend,
    load_shards, make_model,
)
from train_set_aware_evidence_selector import (
    conditional_features, gold_denominators, load_component_cache,
    load_openie_metadata, load_query_plans, normalized_unary_prior, prepare_groups,
)


def state_features(data, group, selected, metadata, mean, std, budget=None, dynamic_plan=False,
                   plan_trace=None):
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("Selected passages must be a nonempty unique sequence")
    mask = group["eligible"].copy()
    mask[selected] = False
    targets = group["labels"][mask]
    if not mask.any():
        return None
    indices = group["indices"]
    plan = query_plan_state_features(
        data["passage_ids"][indices], metadata, group.get("raw_plan", []), selected,
        dynamic=dynamic_plan, static_scores=group["plan_scores"],
        cache=group.setdefault("bound_plan_cache", {}),
        trace=plan_trace,
    )
    references = []
    for reference in selected:
        pair = group["reference_cache"].get(reference)
        if pair is None:
            pair = conditional_features(
                data["base"][indices], group["scores"], reference,
                passage_ids=data["passage_ids"][indices], openie_metadata=metadata,
            )
            group["reference_cache"][reference] = pair
        features = np.concatenate((pair, plan), axis=1)[mask]
        if budget is not None:
            features = append_selection_progress(features, len(selected), budget)
        references.append(((features - mean) / std).astype(np.float32))
    return dict(x=np.stack(references), y=targets,
                candidates=np.flatnonzero(mask),
                prior=normalized_unary_prior(group["scores"], mask),
                selected=list(selected))


def state_logits(model, row, device):
    features = torch.from_numpy(row["x"]).to(device)
    with torch.no_grad():
        logits = model(features).squeeze(-1).amax(0)
    return logits.cpu().numpy() + row["prior"]


def selected_states(data, group, metadata, source_model, source, budget, mode):
    anchor = group["anchor"]
    paths = {"anchor": [[anchor]]}
    if mode == "multistate":
        teacher = [anchor]
        teacher_states = []
        while len(teacher) < budget:
            teacher_states.append(teacher.copy())
            remaining = [int(index) for index in group["order"]
                         if group["labels"][index] and index not in teacher]
            if not remaining:
                break
            teacher.append(remaining[0])
        paths["teacher"] = teacher_states
        mean, std = source["feature_mean"].numpy(), source["feature_std"].numpy()
        policy = [anchor]
        policy_states = []
        while len(policy) < budget:
            policy_states.append(policy.copy())
            row = state_features(data, group, policy, metadata, mean, std)
            if row is None:
                break
            choice = row["candidates"][int(np.argmax(state_logits(source_model, row, "cpu")))]
            policy.append(int(choice))
        paths["frozen_policy"] = policy_states
        wrong = next((int(index) for index in group["order"]
                      if not group["labels"][index] and index != anchor), None)
        if wrong is not None:
            recovery = [anchor, wrong]
            recovery_states = []
            while len(recovery) < budget:
                recovery_states.append(recovery.copy())
                remaining = [int(index) for index in group["order"]
                             if group["labels"][index] and index not in recovery]
                if not remaining:
                    break
                recovery.append(remaining[0])
            paths["recovery"] = recovery_states
    states = {}
    for kind, path in paths.items():
        for selected in path:
            # Set-max scoring is permutation invariant; do not overweight duplicate states.
            key = tuple(sorted(selected))
            if key not in states:
                states[key] = dict(selected=selected, kinds=[])
            states[key]["kinds"].append(kind)
    return list(states.values())


def training_queries(bundle, source, mode, dynamic_plan=False):
    source_model = make_model(51, "cpu")
    source_model.load_state_dict(source["model_state_dict"])
    source_model.eval()
    mean, std = source["feature_mean"].numpy(), source["feature_std"].numpy()
    queries, state_counts, kinds, records = [], Counter(), Counter(), []
    bound_queries, bound_states, bound_variables = set(), 0, 0
    for group in bundle["train_groups"]:
        group["reference_cache"] = {}
        states = selected_states(
            bundle["train"], group, bundle["metadata"], source_model, source,
            min(4, bundle["train_budgets"][group["query_id"]] + 1), mode,
        )
        rows = []
        for state in states:
            trace = {}
            row = state_features(bundle["train"], group, state["selected"],
                                 bundle["metadata"], mean, std, dynamic_plan=dynamic_plan,
                                 plan_trace=trace)
            if row is None or not row["y"].any():
                continue
            rows.append(row)
            if trace.get("bindings"):
                bound_queries.add(group["query_id"])
                bound_states += 1
                bound_variables += len(trace["bindings"])
            state_counts[len(state["selected"])] += 1
            kinds.update(state["kinds"])
            records.append(dict(query_id=group["query_id"], **state,
                                remaining_gold=int(row["y"].sum()),
                                selected_non_gold=int((~group["labels"][state["selected"]]).sum())))
            if dynamic_plan:
                records[-1].update(bound_variables=sorted(trace.get("bindings", {})),
                                   binding_conflicts=trace.get("conflicts", []))
        if rows:
            queries.append(dict(query_id=group["query_id"], states=rows))
        group.pop("reference_cache")
        group.pop("bound_plan_cache", None)
    if not queries:
        raise ValueError("No training queries with remaining gold evidence")
    return queries, dict(queries=len(queries), states=sum(state_counts.values()),
                         selected_count=dict(state_counts), origins=dict(kinds),
                         bound_queries=len(bound_queries), bound_states=bound_states,
                         bound_variables=bound_variables), records


def state_loss_weights(states, reduction="uniform"):
    if not states:
        raise ValueError("Training queries cannot have empty state lists")
    if reduction == "uniform":
        return [1.0 / len(states)] * len(states)
    if reduction != "stage_balanced":
        raise ValueError(f"Unknown state reduction: {reduction}")
    initial = [len(row["selected"]) == 1 for row in states]
    counts = Counter(initial)
    # Equal query mass, with equal mass per present initial/continuation stage.
    return [1.0 / (len(counts) * counts[stage]) for stage in initial]


def training_progress_statistics(queries, budgets, reduction):
    values, weights = [], []
    for query in queries:
        values.extend(selection_progress_values(len(row["selected"]), budgets[query["query_id"]])
                      for row in query["states"])
        weights.extend(state_loss_weights(query["states"], reduction))
    values = np.asarray(values, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float64)
    progress_mean = np.average(values, axis=0, weights=weights).astype(np.float32)
    variance = np.average((values - progress_mean) ** 2, axis=0, weights=weights)
    progress_std = np.maximum(np.sqrt(variance), 1e-6).astype(np.float32)
    progress_std = np.where(progress_std > 1e-6, progress_std, 1.0).astype(np.float32)
    return progress_mean, progress_std


def extend_training_progress(queries, budgets, mean, std, reduction, progress_statistics=None):
    if progress_statistics is None:
        progress_mean, progress_std = training_progress_statistics(queries, budgets, reduction)
    else:
        progress_mean, progress_std = [np.asarray(value, dtype=np.float32)
                                       for value in progress_statistics]
        if (progress_mean.shape != (2,) or progress_std.shape != (2,)
                or not np.isfinite(progress_mean).all() or not np.isfinite(progress_std).all()
                or not (progress_std > 0).all()):
            raise ValueError("Frozen progress statistics require finite two-dimensional mean/std")
    for query in queries:
        budget = budgets[query["query_id"]]
        for row in query["states"]:
            features = append_selection_progress(row["x"], len(row["selected"]), budget)
            features[..., -2:] = (features[..., -2:] - progress_mean) / progress_std
            row["x"] = features
    return np.concatenate((mean, progress_mean)), np.concatenate((std, progress_std))






def make_selector_model(dim, device, seed):
    torch.manual_seed(seed)
    base = make_model(51, device, residual=True)
    if dim == 51:
        return base
    if dim != 53:
        raise ValueError("Selector model supports 51-d or 53-d inputs")
    model = make_model(53, device, residual=True)
    with torch.no_grad():
        model[0].weight[:, :51].copy_(base[0].weight)
        model[0].weight[:, 51:].zero_()
        model[0].bias.copy_(base[0].bias)
        model[2].load_state_dict(base[2].state_dict())
    return model


def multistate_loss(model, queries, device, reduction="uniform"):
    rows, weights = [], []
    for query in queries:
        states = query["states"]
        rows.extend(states)
        weights.extend(state_loss_weights(states, reduction))
    references = max(row["x"].shape[0] for row in rows)
    width = max(len(row["y"]) for row in rows)
    dim = rows[0]["x"].shape[-1]
    x = np.zeros((len(rows), references, width, dim), dtype=np.float32)
    valid_refs = np.zeros((len(rows), references, 1), dtype=bool)
    valid_candidates = np.zeros((len(rows), width), dtype=bool)
    targets = np.zeros_like(valid_candidates)
    prior = np.zeros((len(rows), width), dtype=np.float32)
    for index, row in enumerate(rows):
        ref_count, size = row["x"].shape[:2]
        if not row["y"].any():
            raise ValueError("Each supervised state needs remaining positive evidence")
        x[index, :ref_count, :size] = row["x"]
        valid_refs[index, :ref_count] = True
        valid_candidates[index, :size] = True
        targets[index, :size] = row["y"]
        prior[index, :size] = row["prior"]
    scores = model(torch.from_numpy(x).to(device)).squeeze(-1)
    scores = scores.masked_fill(~torch.from_numpy(valid_refs).to(device), -torch.inf).amax(1)
    scores = (scores + torch.from_numpy(prior).to(device)).masked_fill(
        ~torch.from_numpy(valid_candidates).to(device), -torch.inf)
    y = torch.from_numpy(targets).to(device)
    losses = -torch.log_softmax(scores, 1).masked_fill(~y, 0.0).sum(1) / y.sum(1)
    # Reduce states within each query before averaging across queries.
    return (losses * torch.tensor(weights, device=device)).sum() / len(queries)


def coverage_states(bundle, results):
    groups = {group["query_id"]: group for group in bundle["val_groups"]}
    counts = Counter(zero_hit2=0, complete2=0, complete5=0, complete10=0)
    for ranking in results["rankings"]:
        qid = ranking["query_id"]
        group = groups[qid]
        labels = dict(zip(bundle["val"]["passage_ids"][group["indices"]], group["labels"]))
        hits = [bool(labels[passage]) for passage in ranking["passage_ids"]]
        counts["zero_hit2"] += int(not any(hits[:2]))
        for k in (2, 5, 10):
            counts[f"complete{k}"] += int(sum(hits[:k]) >= bundle["denominators"][qid])
    return dict(counts)


def load_bundle(dataset, cache_root, train_limit, val_limit):
    train_path, val_path = [Path(f"datasets/{dataset}/{split}.json") for split in ("train", "val")]
    query_path = Path(f"outputs/{dataset}/query_triplets_cache.json")
    metadata = load_openie_metadata(f"outputs/{dataset}/openie_results_ner_openai_gpt-oss-120b.json")
    train = load_shards(cache_root / dataset / "train", train_path)
    val = load_component_cache(
        f"outputs/unified_v1/{dataset}/val100_components_seed42.json.gz", val_path, dataset,
        dpr_safety_cache=f"outputs/safety/val/{dataset}/dpr_top200.npz",
        dpr_safety_top_k=200, stream=True,
    )
    for data, limit in ((train, train_limit), (val, val_limit)):
        if limit:
            mask = data["query_ids"] < limit
            for key in data:
                data[key] = data[key][mask]
    train_plans = load_query_plans(train_path, query_path, train["query_ids"])
    plans = load_query_plans(val_path, query_path, val["query_ids"])
    train_groups = prepare_groups(train, None, None, "cpu", metadata,
                                  unary_mode="dpr", query_plans=train_plans)
    val_groups = prepare_groups(val, None, None, "cpu", metadata,
                                unary_mode="dpr", query_plans=plans)
    for data, groups, query_plans in ((train, train_groups, train_plans), (val, val_groups, plans)):
        for group in groups:
            group["raw_plan"] = query_plans[group["query_id"]]
            group["plan_scores"] = query_plan_slot_scores(
                data["passage_ids"][group["indices"]], metadata, query_plans[group["query_id"]])
    return dict(train=train, val=val, train_groups=train_groups, val_groups=val_groups,
                metadata=metadata, plans=plans,
                train_budgets=query_budgets(train_path, query_path, train["query_ids"]),
                budgets=query_budgets(val_path, query_path, val["query_ids"]),
                denominators=gold_denominators(val_path, val["query_ids"]))


def run_dataset(args, dataset, source_root=None, bundle=None):
    started = time.monotonic()
    root = args.root / dataset
    root.mkdir(parents=True, exist_ok=True)
    if (root / "selector_history.json").exists():
        raise ValueError(f"Refusing to overwrite existing history in {root}")
    if bundle is None:
        bundle = load_bundle(dataset, args.cache_root, args.train_limit, args.val_limit)
    if source_root is None:
        source_root = Path("outputs/backend") / dataset
    source_seed = getattr(args, "source_seed", 42)
    source_path = source_root / f"independent_selector_seed{source_seed}.pth"
    tail_path = source_root / f"independent_tail_seed{source_seed}.pth"
    source = torch.load(source_path, map_location="cpu", weights_only=True)
    tail = torch.load(tail_path, map_location="cpu", weights_only=True)
    source = dict(source, bridge_candidate_limit=0)
    _, control_results = evaluate_backend({dataset: bundle}, tail, source)
    baseline = control_results[dataset]
    baseline["coverage_states"] = coverage_states(bundle, baseline)
    queries, counts, records = training_queries(bundle, source, args.mode, args.dynamic_query_plan)
    (root / "training_states.json").write_text(json.dumps(dict(counts=counts, states=records), indent=2))
    print(f"STATES {dataset} {counts}", flush=True)
    # Keep the original training-only normalization for a controlled training-state ablation.
    mean, std = source["feature_mean"].numpy(), source["feature_std"].numpy()
    progress_statistics, progress_hash = None, None
    if args.progress_features:
        if args.progress_normalization_root:
            progress_path = args.progress_normalization_root / dataset / f"multistate_selector_seed{source_seed}.pth"
            progress_checkpoint = torch.load(progress_path, map_location="cpu", weights_only=True)
            if (progress_checkpoint["input_dim"] != 53
                    or not progress_checkpoint.get("uses_selection_progress_features")
                    or progress_checkpoint["training_datasets"] != [dataset]):
                raise ValueError("Frozen progress normalization must come from the same dataset's 53D model")
            frozen_mean, frozen_std = [progress_checkpoint[key].numpy()
                                       for key in ("feature_mean", "feature_std")]
            unchanged = (39 if args.dynamic_query_plan
                         and args.dynamic_plan_normalization == "training" else 51)
            if (not np.array_equal(frozen_mean[:unchanged], mean[:unchanged])
                    or not np.array_equal(frozen_std[:unchanged], std[:unchanged])):
                raise ValueError("Frozen progress model differs in base51 normalization")
            progress_statistics = (frozen_mean[-2:], frozen_std[-2:])
            progress_hash = hashlib.sha256(progress_path.read_bytes()).hexdigest()
        budgets = {qid: min(4, value + 1) for qid, value in bundle["train_budgets"].items()}
        statistics_reduction = getattr(args, "progress_statistics_reduction", None)
        if statistics_reduction:
            if progress_statistics is not None:
                raise ValueError("Choose either frozen or freshly computed training progress statistics")
            progress_statistics = training_progress_statistics(queries, budgets, statistics_reduction)
        mean, std = extend_training_progress(queries, budgets, mean, std, args.state_reduction,
                                            progress_statistics)
    model = make_selector_model(len(mean), args.device, args.seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed)
    history, best_score, best = [], -np.inf, None
    best_values = None
    for epoch in range(args.epochs + 1):
        losses = []
        if epoch:
            model.train()
            order = rng.permutation(len(queries))
            for start in range(0, len(queries), args.batch_size):
                batch = [queries[int(index)] for index in order[start:start + args.batch_size]]
                optimizer.zero_grad(set_to_none=True)
                loss = multistate_loss(model, batch, args.device, args.state_reduction)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
        current = checkpoint(model, mean, std, epoch, selector=True, datasets=(dataset,))
        current.update(training_state_mode=args.mode, state_aggregation="setmax",
                       training_state_source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
                       state_loss_reduction=("mean_stages_then_mean_queries"
                                             if args.state_reduction == "stage_balanced"
                                             else "mean_states_then_mean_queries"),
                       normalization_source="frozen_training_only_single_state",
                       training_budget="min(4,query_budget+1)")
        if args.progress_features:
            current.update(uses_selection_progress_features=True,
                           selection_progress_schema="selected_count_remaining_budget_v1",
                           feature_schema="anchor_bridge_queryplan51_progress2_residual_v1",
                           normalization_source="frozen_base51_plus_training_only_progress2")
            if progress_hash:
                current.update(normalization_source="frozen_base51_plus_frozen_training_progress2",
                               progress_normalization_source_sha256=progress_hash)
        score, values = evaluate_backend({dataset: bundle}, tail, current)
        results = values[dataset]
        results["coverage_states"] = coverage_states(bundle, results)
        history.append(dict(epoch=epoch, score=score, metrics=results["metrics"],
                            coverage_states=results["coverage_states"],
                            loss=float(np.mean(losses)) if losses else None))
        if score > best_score:
            best_score, best, best_values = score, current, results
            torch.save(best, root / f"{args.mode}_selector_seed{args.seed}.pth")
        (root / "selector_history.json").write_text(json.dumps(history, indent=2))
        heads = [round(100 * results["metrics"][f"Recall@{k}"], 2) for k in (2, 5, 10)]
        print(f"EPOCH {dataset} {args.mode} {epoch}/{args.epochs} R2/5/10={heads} "
              f"complete5={results['coverage_states']['complete5']} best={best_score:.6f}", flush=True)
    summary = dict(dataset=dataset, config={key: str(value) if isinstance(value, Path) else value
                                           for key, value in vars(args).items()},
                   agents=f"frozen_seed{getattr(args, 'agent_seed', None) or source_seed}", tail_checkpoint=str(tail_path),
                   selector_source=str(source_path), train_questions=len(bundle["train_groups"]),
                   val_questions=len(bundle["val_groups"]), state_counts=counts,
                   qa=False, repair=False, api_calls=0, best_epoch=best["epoch"],
                   baseline=baseline, validation=best_values,
                   elapsed_seconds=time.monotonic() - started)
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"COMPLETE {dataset} {args.mode} best_epoch={best['epoch']} "
          f"seconds={summary['elapsed_seconds']:.1f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cache_root", type=Path, default=Path("outputs/backend_cache"))
    parser.add_argument("--source_root", type=Path, default=Path("outputs/backend"))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--mode", choices=("single", "multistate"), default="multistate")
    parser.add_argument("--state_reduction", choices=("uniform", "stage_balanced"),
                        default="uniform", help="Uniform states or equal initial/continuation stage mass")
    parser.add_argument("--progress_features", action="store_true",
                        help="Append selected count and remaining budget with training-only normalization")
    parser.add_argument("--dynamic_query_plan", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--ablate_query_plan_features", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--dynamic_plan_normalization", choices=("training", "frozen"),
                        default="training", help="Recalibrate plan12 or preserve original statistics")
    parser.add_argument("--progress_normalization_root", type=Path,
                        help="Reuse same-dataset progress checkpoint statistics for loss-only ablations")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source_seed", type=int, default=42)
    parser.add_argument("--agent_seed", type=int)
    parser.add_argument("--progress_statistics_reduction", choices=("uniform", "stage_balanced"),
                        help="Compute progress normalization from training states with this reduction")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train_limit", type=int, default=0)
    parser.add_argument("--val_limit", type=int, default=0)
    args = parser.parse_args()
    if args.dynamic_query_plan or args.ablate_query_plan_features:
        parser.error("Only fixed RQR is released")
    if args.mode != "multistate" or args.state_reduction != "uniform" or not args.progress_features:
        parser.error("Final training requires multistate, uniform reduction, and progress features")
    if args.batch_size < 1 or min(args.epochs, args.train_limit, args.val_limit) < 0:
        parser.error("Batch size must be positive; limits and epochs nonnegative")
    if len(args.datasets) != len(set(args.datasets)):
        parser.error("Datasets must be unique")
    if args.progress_normalization_root and not args.progress_features:
        parser.error("Frozen progress statistics require --progress_features")
    if args.dynamic_plan_normalization == "frozen" and not args.dynamic_query_plan:
        parser.error("Frozen dynamic normalization requires --dynamic_query_plan")
    if args.ablate_query_plan_features and args.dynamic_query_plan:
        parser.error("QueryPlan removal cannot be combined with dynamic binding")
    torch.set_num_threads(4)
    assert_split_separation(args.datasets)
    for dataset in args.datasets:
        run_dataset(args, dataset, source_root=args.source_root / dataset)


if __name__ == "__main__":
    main()
