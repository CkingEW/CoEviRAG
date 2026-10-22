"""Portable, fixed-protocol CoEviRAG command front end."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
DATASETS = ("2wikimultihopqa", "hotpotqa", "musique", "mrqa_nq_fixed_1k")
STAGES = ("index", "train-policies", "queries", "export", "dense", "safety",
          "train-backend", "train-selector", "retrieve", "qa")


def stage_command(stage, config, split):
    d, seed = config["dataset"], config["seed"]
    if d not in DATASETS or seed != 42:
        raise ValueError("The audited final configuration uses the four primary datasets and seed 42")
    if config["llm_name"] != "openai/gpt-oss-120b":
        raise ValueError("The released reader/OpenIE protocol uses GPT-OSS-120B")
    policy = f"outputs/policies/{d}"
    data = f"datasets/{d}/{split}.json"
    query = f"outputs/{d}/query_triplets_cache.json"
    openie = f"outputs/{d}/openie_results_ner_openai_gpt-oss-120b.json"
    store = f"outputs/{d}/openai_gpt-oss-120b_Qwen_Qwen3-Embedding-4B/chunk_embeddings/vdb_chunk.parquet"
    component = f"outputs/unified_v1/{d}/{split}100{'0' if split == 'test' else ''}_components_seed42.json.gz"
    device = config["device"]
    if stage in {"index", "train-policies"}:
        args = ["scripts/train.py", "--dataset", d, "--llm_name", config["llm_name"],
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
        args = ["scripts/run_frozen_fusion_sweep.py", "--dataset", d, "--eval_split", split,
                "--policy_run_dir", policy, "--llm_base_url", config["llm_base_url"], "--device", device]
        if stage == "queries":
            return args + ["--prepare_queries_only"]
        return args + (["--export_mixed_backend_cache", f"outputs/backend_cache/{d}/train"]
                       if split == "train" else ["--candidate_cache", component])
    if stage == "dense":
        return ["scripts/build_global_dense.py", "--split_data", data, "--embedding_store", store,
                "--output", f"outputs/safety/{split}/{d}/global_dense.json"]
    if stage == "safety":
        return ["scripts/build_dpr_safety_cache.py", "--split_data", data, "--embedding_store", store,
                "--retrieval", f"outputs/safety/{split}/{d}/global_dense.json",
                "--output", f"outputs/safety/{split}/{d}/dpr_top200.npz", "--top_k", "200"]
    if stage == "train-backend":
        return ["scripts/train_mixed_evidence_backend.py", "--root", f"outputs/backend/{d}",
                "--cache_root", "outputs/backend_cache", "--datasets", d,
                "--tail_epochs", str(config["tail_epochs"]), "--selector_epochs", str(config["initial_selector_epochs"])]
    if stage == "train-selector":
        args = ["scripts/train_multistate_evidence_selector.py", "--root", "outputs/selector",
                "--cache_root", "outputs/backend_cache", "--source_root", "outputs/backend",
                "--datasets", d, "--mode", "multistate", "--state_reduction", "uniform",
                "--progress_features", "--epochs", str(config["multistate_epochs"]), "--device", device]
        if config.get("progress_normalization_root"):
            args += ["--progress_normalization_root", config["progress_normalization_root"]]
        return args
    if stage == "retrieve":
        return ["scripts/evaluate_iterative_evidence_selector.py", "--dataset", d, "--split_data", data,
                "--component_cache", component, "--dpr_safety_cache", f"outputs/safety/{split}/{d}/dpr_top200.npz",
                "--dpr_safety_top_k", "200", "--query_cache", query, "--openie_results", openie,
                "--unary_mode", "dpr", "--tail_unary_checkpoint", f"outputs/backend/{d}/independent_tail_seed42.pth",
                "--selector_checkpoint", f"outputs/selector/{d}/multistate_selector_seed42.pth",
                "--bridge_candidate_limit", "0", "--retrieval_mode", "dynamic_setmax_recovery",
                "--stream_components", "--selected_mode_only", "--device", device,
                "--output", f"outputs/retrieval/{d}/{split}_metrics.json",
                "--retrieval_output", f"outputs/retrieval/{d}/{split}.json"]
    if stage == "qa":
        return ["scripts/eval_qa_from_retrieval.py", "--dataset", d, "--split", split,
                "--retrieval_json", f"outputs/retrieval/{d}/{split}.json",
                "--output_json", f"outputs/qa/{d}/{split}_top5.json", "--qa_prompt", "rag_qa_gptoss",
                "--llm_name", config["llm_name"], "--llm_base_url", config["llm_base_url"],
                "--llm_reasoning_effort", "low", "--qa_top_k", "5", "--max_new_tokens", "1024", "--num_workers", "4"]
    raise ValueError("Unknown stage")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/final.json")
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--dry-run", action="store_true", help="Print JSON command only; do not execute the stage")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    allowed = {"dataset", "seed", "device", "llm_name", "llm_base_url", "policy_epochs", "tail_epochs",
               "initial_selector_epochs", "multistate_epochs", "progress_normalization_root"}
    if set(config) - allowed:
        parser.error("Unknown configuration keys are rejected")
    if args.dataset:
        config["dataset"] = args.dataset
    command = [sys.executable, *stage_command(args.stage, config, args.split)]
    if args.dry_run:
        print(json.dumps(dict(stage=args.stage, command=command), indent=2))
        return
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
    environment.setdefault("CUDA_VISIBLE_DEVICES", "1")
    environment.setdefault("PYTHONHASHSEED", "42")
    environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
