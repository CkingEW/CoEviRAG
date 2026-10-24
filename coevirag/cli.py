"""Portable, fixed-protocol CoEviRAG command front end."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
DATASETS = ("2wikimultihopqa", "hotpotqa", "musique", "mrqa_nq_fixed_1k", "sample")
STAGES = ("index", "train-policies", "queries", "export", "dense", "safety",
          "train-backend", "train-selector", "retrieve", "qa")


def workflow_steps(stage, split="test", stage_only=False, with_qa=False):
    if stage_only and with_qa:
        raise ValueError("--stage-only and --with-qa cannot be combined")
    if (stage_only or with_qa) and stage != "retrieve":
        raise ValueError("--stage-only and --with-qa apply only to retrieve")
    if stage == "train":
        return [("index", "train"), ("train-policies", "train"),
                ("queries", "train"), ("queries", "val"),
                ("export", "train"), ("export", "val"),
                ("dense", "val"), ("safety", "val"),
                ("train-backend", "val"), ("train-selector", "val")]
    if stage not in STAGES:
        raise ValueError("Unknown stage")
    if stage == "retrieve":
        if split == "train":
            raise ValueError("Retrieval supports val or test; use export for training features")
        steps = ([("retrieve", split)] if stage_only else
                 [(name, split) for name in ("queries", "export", "dense", "safety", "retrieve")])
        return steps + ([("qa", split)] if with_qa else [])
    return [(stage, split)]


def validate_workflow(stage, config, stage_only=False):
    dataset = config["dataset"]
    if stage == "train":
        histories = [ROOT / "outputs" / dataset / "models/rankers/tail_history.json",
                     ROOT / "outputs" / dataset / "models/selector/selector_history.json"]
        if any(path.exists() for path in histories):
            raise ValueError("Ranking training outputs already exist. Use individual stages to "
                             "continue an interrupted run; move previous outputs before a fresh run.")
    if stage == "retrieve" and not stage_only:
        checkpoints = [ROOT / "outputs" / dataset / "models/agents" / name
                       for name in ("best_policy.pth", "best_paragraph_policy.pth")]
        checkpoints += [ROOT / "outputs" / dataset / "models/rankers/independent_tail_seed42.pth",
                        ROOT / "outputs" / dataset / "models/selector/multistate_selector_seed42.pth"]
        missing = [str(path.relative_to(ROOT)) for path in checkpoints if not path.is_file()]
        if missing:
            raise ValueError("Run train or restore the trained checkpoints before retrieve. Missing: "
                             + ", ".join(missing))


def stage_command(stage, config, split):
    d, seed = config["dataset"], config["seed"]
    if d not in DATASETS or seed != 42:
        raise ValueError("Use a supported benchmark or sample dataset with seed 42")
    if config["llm_name"] != "openai/gpt-oss-120b":
        raise ValueError("The released reader/OpenIE protocol uses GPT-OSS-120B")
    output = f"outputs/{d}"
    cache = f"{output}/cache"
    index = f"{cache}/index"
    models = f"{output}/models"
    results = f"{output}/results"
    policy = f"{models}/agents"
    data = f"datasets/{d}/{split}.json"
    query = f"{index}/query_triplets_cache.json"
    openie = f"{index}/openie_results_ner_openai_gpt-oss-120b.json"
    store = f"{index}/openai_gpt-oss-120b_Qwen_Qwen3-Embedding-4B/chunk_embeddings/vdb_chunk.parquet"
    component = f"{cache}/candidates/{split}.json.gz"
    device = config["device"]
    if stage in {"index", "train-policies"}:
        args = ["scripts/train_agents.py", "--dataset", d, "--llm_name", config["llm_name"],
                "--llm_base_url", config["llm_base_url"], "--llm_reasoning_effort", "low",
                "--max_new_tokens", "1024", "--embedding_name", "Qwen/Qwen3-Embedding-4B",
                "--graph_variant", "unified_v1", "--save_dir", "outputs",
                "--policy_run_dir", policy, "--random_seed", str(seed), "--device", device,
                "--force_index_from_scratch", "false", "--validate_every_n_epochs", "1",
                "--use_ppr_rerank", "false", "--w_ppr", "0", "--w_policy", "0.2",
                "--rerank_fusion_mode", "rank_rrf", "--rank_fusion_rrf_k", "0",
                "--fact_head_protect_k", "0", "--similarity_head_protect_k", "0" if d == "mrqa_nq_fixed_1k" else "2",
                "--retrieval_source_mode", "dual", "--query_fact_score_mode", "concat", "--w_query", "0",
                "--cache_static_retrieval", "true", "--precompute_paragraph_knn", "false",
                "--batch_fact_beam_scoring", "true", "--fact_traversal_mode", "role_aware",
                "--w_fact_policy", "0.25", "--w_paragraph_policy", "0.10", "--w_sim", "0.60",
                "--w_repair_confidence", "0.10", "--w_source_agreement", "0.05",
                "--rl_learning_rate", "0.0001", "--rl_reward_mode", "coverage_progress",
                "--rl_progress_scale", "2", "--rl_terminal_progress_scale", "0.25",
                "--rl_missing_gold_penalty", "0.25", "--rl_complete_evidence_bonus", "0.5",
                "--rl_advantage_mode", "ema_baseline", "--rl_baseline_momentum", "0.95",
                "--rl_entropy_beta", "0.001", "--rl_grad_clip", "1", "--rl_update_batch_episodes", "16",
                "--rl_seed_top_k", "6", "--rl_train_seed_top_k", "6", "--use_paragraph_agent", "true",
                "--use_evidence_paragraph_graph", "true", "--paragraph_train_seed_top_k", "4",
                "--paragraph_train_dpr_top_k", "20", "--paragraph_train_fact_ratio", "0.5",
                "--paragraph_train_fact_seed_source", "policy", "--paragraph_train_fact_search_seed_top_k", "4",
                "--paragraph_train_fact_search_steps", "3", "--paragraph_seed_mode", "hybrid",
                "--paragraph_fact_seed_score_mode", "normalized", "--repair_enabled", "false",
                "--repair_apply_results", "false", "--checkpoint_metric", "mean_recall_2_5_10",
                "--epochs", "0" if stage == "index" else str(config["policy_epochs"]),
                "--train_limit", "1000", "--val_limit", "100", "--early_stopping_patience", "5",
                "--validate_before_training", "true", "--train_fact_agent", "false" if stage == "index" else "true",
                "--train_paragraph_agent", "false" if stage == "index" else "true"]
        if stage == "index":
            args += ["--openie_only", "true", "--save_policy", "false", "--save_training_state", "false"]
        else:
            args += ["--save_policy", "true", "--save_training_state", "true",
                     "--training_state_path", policy + "/latest_training_state.pth"]
        return args
    if stage in {"queries", "export"}:
        args = ["scripts/export_candidates.py", "--dataset", d, "--eval_split", split,
                "--policy_run_dir", policy, "--llm_base_url", config["llm_base_url"], "--device", device]
        if stage == "queries":
            return args + ["--prepare_queries_only"]
        return args + (["--training_cache", f"{cache}/candidates/train"]
                       if split == "train" else ["--candidate_cache", component])
    if stage == "dense":
        return ["scripts/build_global_dense.py", "--split_data", data, "--embedding_store", store,
                "--output", f"{cache}/dense/{split}/global_dense.json"] + (
                    ["--allow_small_corpus"] if d == "sample" else [])
    if stage == "safety":
        return ["scripts/build_dpr_safety_cache.py", "--split_data", data, "--embedding_store", store,
                "--retrieval", f"{cache}/dense/{split}/global_dense.json",
                "--output", f"{cache}/dense/{split}/dpr_top200.npz", "--top_k", "200"] + (
                    ["--allow_small_corpus"] if d == "sample" else [])
    if stage == "train-backend":
        return ["scripts/train_rankers.py", "--output_root", "outputs", "--datasets", d,
                "--tail_epochs", str(config["tail_epochs"]), "--selector_epochs", str(config["initial_selector_epochs"])]
    if stage == "train-selector":
        args = ["scripts/train_selector.py", "--output_root", "outputs",
                "--datasets", d, "--mode", "multistate", "--state_reduction", "uniform",
                "--progress_features", "--epochs", str(config["multistate_epochs"]), "--device", device]
        if config.get("progress_normalization_root"):
            args += ["--progress_normalization_root", config["progress_normalization_root"]]
        return args
    if stage == "retrieve":
        return ["scripts/retrieve.py", "--dataset", d, "--split_data", data,
                "--component_cache", component, "--dpr_safety_cache", f"{cache}/dense/{split}/dpr_top200.npz",
                "--dpr_safety_top_k", "200", "--query_cache", query, "--openie_results", openie,
                "--unary_mode", "dpr", "--tail_unary_checkpoint", f"{models}/rankers/independent_tail_seed42.pth",
                "--selector_checkpoint", f"{models}/selector/multistate_selector_seed42.pth",
                "--bridge_candidate_limit", "0", "--retrieval_mode", "dynamic_setmax_recovery",
                "--stream_components", "--selected_mode_only", "--device", device,
                "--output", f"{results}/retrieval/{split}_metrics.json",
                "--retrieval_output", f"{results}/retrieval/{split}.json"]
    if stage == "qa":
        return ["scripts/generate_answers.py", "--dataset", d, "--split", split,
                "--retrieval_json", f"{results}/retrieval/{split}.json",
                "--output_json", f"{results}/qa/{split}_top5.json", "--qa_prompt", "rag_qa_gptoss",
                "--llm_name", config["llm_name"], "--llm_base_url", config["llm_base_url"],
                "--llm_reasoning_effort", "low", "--qa_top_k", "5", "--max_new_tokens", "1024", "--num_workers", "4"]
    raise ValueError("Unknown stage")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", *STAGES),
                        help="train runs the full training workflow; retrieve prepares caches and ranks passages")
    parser.add_argument("--config", type=Path,
                        help="Override configuration (default: sample.json for --dataset sample, otherwise final.json)")
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--split", choices=("train", "val", "test"),
                        help="Question split (default: test); train always uses train and val")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--stage-only", action="store_true",
                       help="For retrieve, rank existing caches without rebuilding candidates")
    modes.add_argument("--with-qa", action="store_true",
                       help="For retrieve, also generate answers and evaluate QA")
    parser.add_argument("--dry-run", action="store_true", help="Print JSON commands without executing them")
    args = parser.parse_args()
    config_path = args.config or ROOT / "configs" / (
        "sample.json" if args.dataset == "sample" else "final.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    allowed = {"dataset", "seed", "device", "llm_name", "llm_base_url", "policy_epochs", "tail_epochs",
               "initial_selector_epochs", "multistate_epochs", "progress_normalization_root"}
    if set(config) - allowed:
        parser.error("Unknown configuration keys are rejected")
    if args.dataset:
        config["dataset"] = args.dataset
    if args.stage == "train" and args.split is not None:
        parser.error("train uses both train and val; omit --split")
    try:
        steps = workflow_steps(args.stage, args.split or "test", args.stage_only, args.with_qa)
        commands = [dict(stage=stage, split=split,
                         command=[sys.executable, *stage_command(stage, config, split)])
                    for stage, split in steps]
    except ValueError as error:
        parser.error(str(error))
    if args.dry_run:
        result = (dict(stage=args.stage, command=commands[0]["command"]) if len(commands) == 1
                  else dict(stage=args.stage, dataset=config["dataset"], steps=commands))
        print(json.dumps(result, indent=2))
        return
    try:
        validate_workflow(args.stage, config, args.stage_only)
    except ValueError as error:
        parser.error(str(error))
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
    environment.setdefault("CUDA_VISIBLE_DEVICES", "1")
    environment.setdefault("PYTHONHASHSEED", "42")
    environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    for index, step in enumerate(commands, 1):
        print(f"[{index}/{len(commands)}] {step['stage']} ({step['split']})", flush=True)
        subprocess.run(step["command"], cwd=ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
