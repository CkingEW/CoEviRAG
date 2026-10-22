# Modified for CoEviRAG: reject unsupported selector checkpoint protocols.
import argparse
import json
from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from train_set_aware_evidence_selector import (
    evaluate,
    gold_denominators,
    load_npz_cache,
    load_openie_metadata,
    load_unary_ranker,
    prepare_groups,
)


def load_selector(path, device, bridge_candidate_limit=None):
    if bridge_candidate_limit is not None and bridge_candidate_limit < 0:
        raise ValueError("bridge_candidate_limit must be nonnegative")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    from coevirag.release_protocol import validate_selector
    validate_selector(checkpoint)
    if bridge_candidate_limit is not None:
        checkpoint = {**checkpoint, "bridge_candidate_limit": bridge_candidate_limit}
    model = torch.nn.Sequential(
        torch.nn.Linear(int(checkpoint["input_dim"]), int(checkpoint["hidden_dim"])),
        torch.nn.ReLU(),
        torch.nn.Linear(int(checkpoint["hidden_dim"]), 1),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    schema = str(checkpoint.get("feature_schema", ""))
    selector_mode = "residual" if "residual" in schema else "direct"
    return model, checkpoint, selector_mode


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate frozen anchor-bridge evidence selectors."
    )
    parser.add_argument("--candidate_cache", required=True)
    parser.add_argument("--split_data", required=True)
    parser.add_argument("--openie_results", required=True)
    parser.add_argument("--unary_checkpoint", required=True)
    parser.add_argument("--selector_checkpoints", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device.startswith("cuda") and torch.cuda.is_available()
        else "cpu"
    )
    data = load_npz_cache(args.candidate_cache)
    metadata = load_openie_metadata(args.openie_results)
    unary_model, unary_checkpoint = load_unary_ranker(
        args.unary_checkpoint, device
    )
    denominators = gold_denominators(args.split_data, data["query_ids"])
    results = []
    for path in args.selector_checkpoints:
        model, checkpoint, selector_mode = load_selector(path, device)
        groups = prepare_groups(
            data, unary_model, unary_checkpoint, device, metadata,
            int(checkpoint.get("bridge_candidate_limit", 0)),
        )
        expected_dim = groups[0]["pair_features"].shape[1]
        if int(checkpoint["input_dim"]) != expected_dim:
            raise ValueError(
                f"Selector {path} expects {checkpoint['input_dim']} features, "
                f"but evaluation data builds {expected_dim}"
            )
        result = evaluate(
            groups,
            model,
            checkpoint["feature_mean"].cpu().numpy(),
            checkpoint["feature_std"].cpu().numpy(),
            denominators,
            selector_mode,
            device,
        )
        results.append({"checkpoint": path, **result})

    payload = {
        "candidate_cache": args.candidate_cache,
        "queries": len(set(data["query_ids"].tolist())),
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    concise = {
        result["checkpoint"]: {
            "baseline": result["baseline"]["metrics"],
            "set_aware": result["set_aware"]["metrics"],
            "mean_recall_2_5_10": result["set_aware"]["mean_recall_2_5_10"],
            "states": result["states"],
        }
        for result in results
    }
    print(json.dumps(concise, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
