#!/usr/bin/env python3
# Modified for CoEviRAG: enforce fixed-RQR final ranking and checkpoint schema.
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_set_aware_evidence_selector import load_selector
from train_set_aware_evidence_selector import (
    RECALL_KS,
    conditional_features,
    gold_denominators,
    load_npz_cache,
    load_component_cache,
    load_openie_metadata,
    load_query_plans,
    load_unary_ranker,
    normalized_unary_prior,
    unary_scores,
)
from coevirag.query_plan_evidence import (
    append_selection_progress, dynamic_query_plan_enabled, query_plan_slot_scores,
    query_plan_state_features,
    mask_query_plan_features,
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


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate fixed and QueryPlan-budget iterative evidence selection."
    )
    cache_group = parser.add_mutually_exclusive_group(required=True)
    cache_group.add_argument("--candidate_cache")
    cache_group.add_argument("--component_cache")
    parser.add_argument("--dpr_safety_cache", default="")
    parser.add_argument("--dpr_safety_top_k", type=int, default=0)
    parser.add_argument("--stream_components", action="store_true")
    parser.add_argument("--selected_mode_only", action="store_true",
                        help="Evaluate only the frozen export mode, without other configurations")
    parser.add_argument("--repair_top_k", type=int, default=0)
    parser.add_argument("--repair_reference_cache", default="")
    parser.add_argument("--dataset", default="hotpotqa")
    parser.add_argument("--split_data", required=True)
    parser.add_argument("--query_cache", required=True)
    parser.add_argument("--openie_results", required=True)
    parser.add_argument("--unary_checkpoint", default="")
    parser.add_argument(
        "--tail_unary_checkpoint", default="",
        help=(
            "Optional learned unary ranker used only after the evidence-selection "
            "budget. The anchor and selector still use --unary_mode."
        ),
    )
    parser.add_argument(
        "--unary_mode", choices=("auto", "learned", "dpr"), default="auto"
    )
    parser.add_argument("--selector_checkpoint", required=True)
    parser.add_argument("--bridge_candidate_limit", type=int, default=None,
                        help="Evaluation override: 0 considers the full candidate pool; default uses checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--retrieval_output")
    parser.add_argument(
        "--retrieval_mode",
        default="dynamic_setmax",
        help="Ranking configuration to export when --retrieval_output is set.",
    )
    parser.add_argument(
        "--extra_budgets", type=int, nargs="*", default=[],
        help="Also evaluate set-max selection with these fixed budgets.",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if (args.unary_mode != "dpr" or args.retrieval_mode != "dynamic_setmax_recovery"
            or not args.selected_mode_only or not args.tail_unary_checkpoint
            or not args.dpr_safety_cache or args.dpr_safety_top_k != 200
            or args.bridge_candidate_limit != 0 or args.repair_top_k
            or args.repair_reference_cache or args.extra_budgets):
        parser.error("Use the CoEviRAG CLI for locked final retrieval (DPR200, 53D prefix, independent tail)")

    device = torch.device(
        args.device if args.device.startswith("cuda") and torch.cuda.is_available()
        else "cpu"
    )
    data = (
        load_npz_cache(args.candidate_cache)
        if args.candidate_cache else
        load_component_cache(
            args.component_cache,
            args.split_data,
            args.dataset,
            args.dpr_safety_cache,
            args.dpr_safety_top_k,
            args.repair_top_k,
            args.repair_reference_cache,
            stream=args.stream_components,
        )
    )
    metadata = load_openie_metadata(args.openie_results)
    selector, checkpoint, _ = load_selector(
        args.selector_checkpoint, device, args.bridge_candidate_limit
    )
    from coevirag.release_protocol import validate_selector
    validate_selector(checkpoint, final=True)
    unary_mode = (
        str(checkpoint.get("unary_mode", "learned"))
        if args.unary_mode == "auto" else args.unary_mode
    )
    if unary_mode == "learned":
        if not args.unary_checkpoint:
            raise ValueError("--unary_checkpoint is required for learned unary mode")
        unary_model, unary_checkpoint = load_unary_ranker(
            args.unary_checkpoint, device
        )
    else:
        unary_model, unary_checkpoint = None, None
    if args.tail_unary_checkpoint:
        tail_unary_model, tail_unary_checkpoint = load_unary_ranker(
            args.tail_unary_checkpoint, device
        )
        if tail_unary_checkpoint.get("input_dim") != 60:
            raise ValueError("The independent tail must use the adopted 60D schema")
    else:
        tail_unary_model, tail_unary_checkpoint = None, None
    denominators = gold_denominators(args.split_data, data["query_ids"])
    dynamic_budgets = query_budgets(
        args.split_data, args.query_cache, data["query_ids"]
    )
    query_plans = load_query_plans(
        args.split_data, args.query_cache, data["query_ids"]
    )
    configurations = {
        "select2": ("last", 2),
        "select3_chain": ("last", 3),
        "select4_chain": ("last", 4),
        "dynamic_chain": ("last", None),
        "dynamic_setmax": ("setmax", None),
        "dynamic_setmax_recovery": ("setmax", -1),
    }
    for budget in args.extra_budgets:
        if budget < 2:
            raise ValueError("--extra_budgets values must be at least 2")
        configurations[f"select{budget}_setmax"] = ("setmax", budget)
    valid_retrieval_modes = {"baseline", *configurations}
    if args.retrieval_mode not in valid_retrieval_modes:
        raise ValueError(
            f"Unknown --retrieval_mode {args.retrieval_mode!r}; expected one of "
            f"{sorted(valid_retrieval_modes)}"
        )
    if args.selected_mode_only:
        configurations = {name: value for name, value in configurations.items()
                          if name == args.retrieval_mode}
    values = {name: [] for name in ("baseline", *configurations)}
    complete = {name: 0 for name in configurations}
    samples = json.loads(Path(args.split_data).read_text(encoding="utf-8"))
    retrieval_results = []

    for query_id in np.unique(data["query_ids"]):
        indices = np.flatnonzero(data["query_ids"] == query_id)
        base = data["base"][indices]
        labels = data["labels"][indices]
        passage_ids = data["passage_ids"][indices]
        scores = (
            data["base"][indices, 0].astype(np.float32)
            if unary_mode == "dpr" else
            unary_scores(
                data["features"][indices], unary_model, unary_checkpoint, device
            )
        )
        tail_scores = (
            unary_scores(
                data["features"][indices], tail_unary_model,
                tail_unary_checkpoint, device,
            )
            if tail_unary_model is not None else None
        )
        denominator = denominators[int(query_id)]
        plan_scores = (
            query_plan_slot_scores(
                passage_ids, metadata, query_plans[int(query_id)]
            )
            if checkpoint.get("uses_query_plan_features", False) else None
        )
        baseline = np.argsort(-scores, kind="stable")
        values["baseline"].append([
            float(labels[baseline[:k]].sum()) / denominator for k in RECALL_KS
        ])
        selected_retrieval_ranking = baseline if args.retrieval_mode == "baseline" else None
        for name, (mode, fixed_budget) in configurations.items():
            if fixed_budget == -1:
                budget = min(4, dynamic_budgets[int(query_id)] + 1)
            elif fixed_budget is None:
                budget = dynamic_budgets[int(query_id)]
            else:
                budget = fixed_budget
            ranking = select_ranking(
                base, passage_ids, scores, selector, checkpoint, mode, budget,
                metadata, device, tail_scores=tail_scores,
                raw_query_triples=query_plans[int(query_id)],
                query_plan_scores=plan_scores,
            )
            values[name].append([
                float(labels[ranking[:k]].sum()) / denominator for k in RECALL_KS
            ])
            complete[name] += int(labels[ranking[:budget]].sum() >= denominator)
            if name == args.retrieval_mode:
                selected_retrieval_ranking = ranking

        if args.retrieval_output:
            if selected_retrieval_ranking is None:
                raise RuntimeError(
                    f"No ranking produced for retrieval mode {args.retrieval_mode}"
                )
            ranked_passage_ids = passage_ids[selected_retrieval_ranking]
            retrieval_results.append({
                "question": samples[int(query_id)]["question"],
                "docs": [metadata[str(passage_id)]["content"] for passage_id in ranked_passage_ids],
                "scores": [float(scores[index]) for index in selected_retrieval_ranking],
            })

    payload = {
        "config": vars(args),
        "queries": int(len(np.unique(data["query_ids"]))),
        "dynamic_budget_counts": {
            str(k): int(sum(value == k for value in dynamic_budgets.values()))
            for k in sorted(set(dynamic_budgets.values()))
        },
        "results": {
            name: {**summarize(result), "complete_at_selection_budget": complete.get(name)}
            for name, result in values.items()
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.retrieval_output:
        retrieval_output = Path(args.retrieval_output)
        retrieval_output.parent.mkdir(parents=True, exist_ok=True)
        retrieval_output.write_text(json.dumps({
            "metadata": {
                "dataset": args.dataset,
                "split_data": args.split_data,
                "ranking_mode": args.retrieval_mode,
                "unary_mode": unary_mode,
                "dpr_safety_cache": args.dpr_safety_cache,
                "dpr_safety_top_k": args.dpr_safety_top_k,
                "bridge_candidate_limit": int(checkpoint.get("bridge_candidate_limit", 0)),
                "metrics": payload["results"][args.retrieval_mode]["metrics"],
            },
            "results": retrieval_results,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        name: {
            "metrics": result["metrics"],
            "mean_recall_2_5_10": result["mean_recall_2_5_10"],
            "complete_at_selection_budget": result["complete_at_selection_budget"],
        }
        for name, result in payload["results"].items()
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
