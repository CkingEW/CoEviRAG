"""Construct the dual graphs and train Fact and Passage navigation agents."""
import os
import json
import argparse
import logging
import math
import random
import time
import numpy as np
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import torch
import sys
from pathlib import Path


project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from coevirag.CoEviRAG import CoEviRAG
from coevirag.utils.config_utils import BaseConfig
from coevirag.utils.misc_utils import string_to_bool, min_max_normalize


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def save_retrieval_rankings(
        output_path: str,
        retrieval_results,
        retrieval_metrics: Dict[str, float],
        metadata: Dict):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata,
        "metrics": retrieval_metrics,
        "results": [
            {
                "question": result.question,
                "docs": result.docs,
                "scores": [float(score) for score in result.doc_scores],
            }
            for result in retrieval_results
        ],
    }
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False)
    logger.info("Saved per-query CoEviRAG rankings to %s", path)


def get_checkpoint_score(metrics: Dict[str, float], metric_name: str) -> float:
    if metric_name == "recall10":
        return float(metrics.get("Recall@10", 0.0))
    if metric_name == "mean_recall_2_5_10":
        return float(np.mean([
            metrics.get("Recall@2", 0.0),
            metrics.get("Recall@5", 0.0),
            metrics.get("Recall@10", 0.0),
        ]))
    raise ValueError(f"Unsupported checkpoint metric: {metric_name}")


def checkpoint_decision(current_score, best_score, progress_score, min_delta=0.0):
    """Prefer the latest tied checkpoint without treating ties as progress."""
    if not math.isfinite(current_score):
        return False, False
    replace = current_score >= best_score
    improved = current_score > progress_score + max(1e-12, min_delta)
    return replace, improved


def save_training_state(
        path: str,
        epoch: int,
        coevirag: CoEviRAG,
        best_checkpoint_score: float,
        best_val_metrics: Dict[str, float],
        best_epoch: int,
        checkpoint_metric: str,
        early_stopping_best_score: Optional[float] = None,
        validation_checks_without_improvement: int = 0) -> None:
    state = {
        "epoch": epoch,
        "checkpoint_metric": checkpoint_metric,
        "best_checkpoint_score": best_checkpoint_score,
        "best_val_metrics": dict(best_val_metrics),
        "best_epoch": best_epoch,
        "checkpoint_tie_break": "latest_epoch",
        "early_stopping_best_score": (
            best_checkpoint_score if early_stopping_best_score is None
            else early_stopping_best_score
        ),
        "validation_checks_without_improvement": validation_checks_without_improvement,
        "fact_policy": coevirag.rl_policy_network.state_dict(),
        "paragraph_policy": coevirag.paragraph_policy_network.state_dict(),
        "fact_optimizer": coevirag.rl_agent.optimizer.state_dict(),
        "paragraph_optimizer": coevirag.paragraph_agent.optimizer.state_dict(),
        "fact_return_baseline": coevirag.rl_agent.return_baseline,
        "fact_baseline_initialized": coevirag.rl_agent.baseline_initialized,
        "paragraph_return_baseline": coevirag.paragraph_agent.return_baseline,
        "paragraph_baseline_initialized": coevirag.paragraph_agent.baseline_initialized,
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(state, path)


def load_training_state(path: str, coevirag: CoEviRAG, checkpoint_metric: str):
    try:
        state = torch.load(path, map_location=coevirag.device, weights_only=False)
    except TypeError:
        state = torch.load(path, map_location=coevirag.device)

    saved_metric = state.get("checkpoint_metric", checkpoint_metric)
    if saved_metric != checkpoint_metric:
        raise ValueError(
            f"Training state uses checkpoint metric {saved_metric!r}, "
            f"but this run requested {checkpoint_metric!r}."
        )

    coevirag.rl_policy_network.load_state_dict(state["fact_policy"], strict=False)
    coevirag.paragraph_policy_network.load_state_dict(
        state["paragraph_policy"],
        strict=False,
    )
    coevirag.rl_agent.optimizer.load_state_dict(state["fact_optimizer"])
    coevirag.paragraph_agent.optimizer.load_state_dict(state["paragraph_optimizer"])
    coevirag.rl_agent.return_baseline = float(state.get("fact_return_baseline", 0.0))
    coevirag.rl_agent.baseline_initialized = bool(
        state.get("fact_baseline_initialized", False)
    )
    coevirag.paragraph_agent.return_baseline = float(
        state.get("paragraph_return_baseline", 0.0)
    )
    coevirag.paragraph_agent.baseline_initialized = bool(
        state.get("paragraph_baseline_initialized", False)
    )

    random.setstate(state["python_rng_state"])
    np.random.set_state(state["numpy_rng_state"])
    torch.set_rng_state(state["torch_rng_state"].cpu())
    cuda_rng_state = state.get("cuda_rng_state")
    if torch.cuda.is_available() and cuda_rng_state is not None:
        for device_idx, rng_state in enumerate(
                cuda_rng_state[:torch.cuda.device_count()]):
            torch.cuda.set_rng_state(rng_state.cpu(), device=device_idx)
    return state


def get_gold_docs(samples: List[Dict], dataset_name: str = None) -> List[List[str]]:
    gold_docs = []
    for sample in samples:
        doc_set = set()
        if 'supporting_facts' in sample:
            gold_titles = set(item[0] for item in sample['supporting_facts'])
            for item in sample['context']:
                if item[0] in gold_titles:
                    text = ''.join(item[1]) if dataset_name and dataset_name.startswith('hotpotqa') else ' '.join(item[1])
                    doc_set.add(item[0] + '\n' + text)
        elif 'contexts' in sample:
            for item in sample['contexts']:
                if item['is_supporting']:
                    doc_set.add(item['title'] + '\n' + item['text'])
        elif 'paragraphs' in sample:
            for p in sample['paragraphs']:
                if p.get('is_supporting', True):
                    doc_set.add(p['title'] + '\n' + p.get('text', p.get('paragraph_text', '')))
        else:
            logger.warning(f"Sample missing valid answer source fields: {sample.get('id', 'N/A')}")
        gold_docs.append(list(doc_set))
    return gold_docs

def precompute_gold_embeddings(coevirag: CoEviRAG, all_data_points: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Precompute embeddings for all gold documents in the training/validation sets.
    """
    logger.info("Precomputing embeddings for all gold documents...")
    
    # 1. Collect all unique document texts
    unique_gold_texts = set()
    for item in all_data_points:
        # item['gold_docs'] is a list of strings
        for doc_text in item['gold_docs']:
            unique_gold_texts.add(doc_text)
            
    sorted_texts = list(unique_gold_texts)
    logger.info(f"Total of {len(sorted_texts)} unique gold documents to encode.")
    
    if not sorted_texts:
        return {}

    # 2. Batch encode
    # No instruction needed for documents, or use empty string
    embeddings_np = coevirag.embedding_model.batch_encode(
        sorted_texts, 
        instruction="", 
        norm=True
    )
    
    # 3. Build lookup dictionary
    cache = {}
    for text, emb in zip(sorted_texts, embeddings_np):

        cache[text] = torch.from_numpy(emb).float()
        
    logger.info("Gold document embedding precomputation completed.")
    return cache


def paragraph_evidence_utility(
        coevirag: CoEviRAG,
        passage_info: Dict[str, Dict],
        gold_docs: List[str],
        complete_weight: float) -> float:
    """Score one query using mean Recall/CompleteEvidence at 2, 5, and 10."""
    gold_set = set(gold_docs or [])
    if not gold_set:
        return 0.0
    ranked_passages = sorted(
        passage_info,
        key=lambda passage: (
            passage_info[passage].get("score", -float("inf")),
            passage,
        ),
        reverse=True,
    )
    ranked_contents = []
    for passage in ranked_passages[:10]:
        try:
            ranked_contents.append(
                coevirag.chunk_embedding_store.get_row(passage)["content"]
            )
        except Exception:
            continue

    recalls = []
    complete = []
    for top_k in (2, 5, 10):
        hits = gold_set.intersection(ranked_contents[:top_k])
        recalls.append(len(hits) / len(gold_set))
        complete.append(float(len(hits) == len(gold_set)))
    recall_weights = np.asarray([
        max(0.0, float(getattr(
            coevirag.global_config,
            "rl_downstream_recall2_weight",
            1.0,
        ))),
        max(0.0, float(getattr(
            coevirag.global_config,
            "rl_downstream_recall5_weight",
            1.0,
        ))),
        max(0.0, float(getattr(
            coevirag.global_config,
            "rl_downstream_recall10_weight",
            1.0,
        ))),
    ], dtype=np.float32)
    weight_sum = float(recall_weights.sum())
    if weight_sum <= 0.0:
        raise ValueError("At least one downstream Recall weight must be positive")
    recall_weights /= weight_sum
    return float(
        np.dot(recall_weights, np.asarray(recalls, dtype=np.float32))
        + complete_weight
        * np.dot(recall_weights, np.asarray(complete, dtype=np.float32))
    )


def fact_path_to_paragraph_seeds(
        coevirag: CoEviRAG,
        query: str,
        path: List[str],
        max_seeds: int) -> List[Tuple[str, float]]:
    """Map a sampled Fact path to query-scored supporting paragraph seeds."""
    query_emb = coevirag._get_query_embedding_tensor(query).view(1, -1)
    seed_scores = {}
    for node_key in path:
        for passage_key in coevirag.get_supporting_passages_for_node(node_key):
            if passage_key not in coevirag.passage_key_to_idx:
                continue
            if passage_key in getattr(coevirag, "base_retrieval_masked_passages", set()):
                continue
            passage_emb = coevirag.get_embedding_for_node(passage_key)
            if passage_emb is None:
                continue
            score = torch.nn.functional.cosine_similarity(
                query_emb,
                passage_emb.view(1, -1),
                dim=1,
            ).item()
            seed_scores[passage_key] = max(seed_scores.get(passage_key, -1.0), score)
    ranked = sorted(
        seed_scores.items(),
        key=lambda item: (item[1], item[0]),
        reverse=True,
    )[:max(1, int(max_seeds))]
    return [(passage, float((score + 1.0) / 2.0)) for passage, score in ranked]


def paragraph_downstream_gain(
        coevirag: CoEviRAG,
        query: str,
        gold_docs: List[str],
        path: List[str]) -> Tuple[float, float, float, int]:
    """Compare frozen Paragraph utility with DPR-only and DPR+Fact seeds."""
    complete_weight = float(getattr(
        coevirag.global_config,
        "rl_downstream_complete_weight",
        0.25,
    ))
    baseline_cache = getattr(coevirag, "_downstream_dpr_utility_cache", None)
    if baseline_cache is None:
        baseline_cache = {}
        coevirag._downstream_dpr_utility_cache = baseline_cache

    paragraph_was_training = coevirag.paragraph_policy_network.training
    coevirag.paragraph_policy_network.eval()
    with torch.no_grad():
        if query not in baseline_cache:
            baseline_info = coevirag.run_paragraph_graph_search(
                query=query,
                seed_passages=[],
            )
            baseline_cache[query] = paragraph_evidence_utility(
                coevirag,
                baseline_info,
                gold_docs,
                complete_weight,
            )
        baseline_utility = float(baseline_cache[query])
        fact_seeds = fact_path_to_paragraph_seeds(
            coevirag=coevirag,
            query=query,
            path=path,
            max_seeds=getattr(
                coevirag.global_config,
                "rl_downstream_seed_top_k",
                6,
            ),
        )
        if fact_seeds:
            seeded_info = coevirag.run_paragraph_graph_search(
                query=query,
                seed_passages=fact_seeds,
            )
            seeded_utility = paragraph_evidence_utility(
                coevirag,
                seeded_info,
                gold_docs,
                complete_weight,
            )
        else:
            seeded_utility = baseline_utility
    if paragraph_was_training:
        coevirag.paragraph_policy_network.train()
    return (
        float(seeded_utility - baseline_utility),
        baseline_utility,
        float(seeded_utility),
        len(fact_seeds),
    )


def prepare_datasets(dataset_name, base_dir, val_split, test_split, random_seed=42):
    paths = [Path(base_dir) / (split + ".json") for split in ("train", "val", "test")]
    if not all(path.is_file() for path in paths):
        raise FileNotFoundError("Prepare ordered manifest splits with scripts/prepare_data.py first")
    return tuple(json.loads(path.read_text(encoding="utf-8")) for path in paths[:2])


def train_rl_agent_epoch(coevirag: CoEviRAG, training_data: List[Dict], epoch_num: int, gold_emb_cache: Dict[str, torch.Tensor]):
    agent = coevirag.rl_agent
    total_epoch_reward = 0
    total_episodes = 0
    completed_start_skips = 0
    downstream_enabled = bool(getattr(
        coevirag.global_config,
        "rl_hierarchical_downstream_reward",
        False,
    ))
    downstream_weight = float(getattr(
        coevirag.global_config,
        "rl_downstream_reward_weight",
        1.0,
    ))
    downstream_clip = max(0.0, float(getattr(
        coevirag.global_config,
        "rl_downstream_reward_clip",
        1.0,
    )))
    downstream_credit_mode = str(getattr(
        coevirag.global_config,
        "rl_downstream_credit_mode",
        "raw",
    ))
    if downstream_credit_mode not in {"raw", "group_relative"}:
        raise ValueError(
            f"Unsupported downstream credit mode: {downstream_credit_mode}"
        )
    downstream_positive_bonus = max(0.0, float(getattr(
        coevirag.global_config,
        "rl_downstream_positive_bonus",
        0.0,
    )))
    downstream_recall_weights = (
        float(getattr(coevirag.global_config, "rl_downstream_recall2_weight", 1.0)),
        float(getattr(coevirag.global_config, "rl_downstream_recall5_weight", 1.0)),
        float(getattr(coevirag.global_config, "rl_downstream_recall10_weight", 1.0)),
    )
    seed_gate_enabled = bool(getattr(
        coevirag.global_config,
        "rl_paragraph_seed_gate_enabled",
        False,
    ))
    seed_gate_cost = max(0.0, float(getattr(
        coevirag.global_config,
        "rl_seed_gate_cost",
        0.01,
    )))
    seed_gate_loss_weight = max(0.0, float(getattr(
        coevirag.global_config,
        "rl_seed_gate_loss_weight",
        5.0,
    )))
    if seed_gate_enabled and not downstream_enabled:
        raise ValueError("Fact seed gate training requires hierarchical downstream reward")
    downstream_deltas = []
    downstream_credits = []
    downstream_baselines = []
    downstream_seeded = []
    downstream_seed_counts = []
    seed_gate_advantages = []
    seed_gate_probabilities = []
    pending_losses = []
    update_batch_episodes = max(
        1,
        int(getattr(coevirag.global_config, "rl_update_batch_episodes", 1)),
    )
    progress_bar = tqdm(training_data, desc=f"Epoch {epoch_num} [Training]")
    
    # Ensure model is in training mode
    coevirag.rl_policy_network.train()

    for data_point in progress_bar:
        query, gold_docs = data_point['query'], data_point['gold_docs']

        # --- Get Gold Doc Embeddings from Cache ---
        gold_doc_embs = None
        if gold_docs:
            embs_list = []
            for doc in gold_docs:
                if doc in gold_emb_cache:
                    embs_list.append(gold_emb_cache[doc])
            
            if embs_list:
                gold_doc_embs = torch.stack(embs_list) # Stack into (N, dim)
        
        # --- Get Seed Nodes ---
        query_fact_scores = coevirag.get_fact_scores(query)
        top_k_fact_indices, top_k_facts = coevirag.rank_facts(query, query_fact_scores)
        ranked_seeds = coevirag.get_ranked_seed_nodes(
            top_k_facts=top_k_facts,
            top_k_fact_indices=top_k_fact_indices,
            query_fact_scores=query_fact_scores,
            max_seeds=getattr(coevirag.global_config, "rl_train_seed_top_k", 3),
            include_fact_nodes=True,
        )
        if not ranked_seeds:
            continue
        
        # --- Run top-k seed episodes and update ---
        query_episode_records = []
        for start_node, _ in ranked_seeds:
            path = agent.run_episode(
                coevirag_instance=coevirag,
                start_node=start_node,
                query=query,
                gold_docs=gold_docs,
                gold_doc_embs=gold_doc_embs,
                max_steps=coevirag.rl_max_episode_steps,
                reward_gold=coevirag.rl_reward_gold,
                reward_step=coevirag.rl_reward_step
            )

            if getattr(agent, "last_episode_start_skipped", False):
                completed_start_skips += 1
                continue
            if downstream_enabled:
                trace = agent.take_episode_trace()
                downstream_result = None
                if trace["rewards"]:
                    downstream_result = paragraph_downstream_gain(
                        coevirag=coevirag,
                        query=query,
                        gold_docs=gold_docs,
                        path=path,
                    )
                query_episode_records.append({
                    "trace": trace,
                    "downstream_result": downstream_result,
                    "start_node": start_node,
                })
            else:
                total_epoch_reward += sum(agent.rewards)
                total_episodes += 1
                policy_loss = agent.consume_episode_loss()
                if policy_loss is not None:
                    pending_losses.append(policy_loss)
                if len(pending_losses) >= update_batch_episodes:
                    agent.apply_policy_losses(pending_losses)
                    pending_losses = []

        if downstream_enabled:
            valid_deltas = [
                record["downstream_result"][0]
                for record in query_episode_records
                if record["downstream_result"] is not None
            ]
            query_delta_mean = float(np.mean(valid_deltas)) if valid_deltas else 0.0
            for record in query_episode_records:
                trace = record["trace"]
                result = record["downstream_result"]
                credit = 0.0
                if result is not None:
                    delta, baseline_utility, seeded_utility, seed_count = result
                    if downstream_credit_mode == "group_relative":
                        credit = (
                            delta
                            - query_delta_mean
                            + downstream_positive_bonus * max(0.0, delta)
                        )
                    else:
                        credit = delta
                    if downstream_clip > 0:
                        credit = float(np.clip(
                            credit,
                            -downstream_clip,
                            downstream_clip,
                        ))
                    downstream_deltas.append(delta)
                    downstream_credits.append(credit)
                    downstream_baselines.append(baseline_utility)
                    downstream_seeded.append(seeded_utility)
                    downstream_seed_counts.append(seed_count)

                    if seed_gate_enabled:
                        max_gate_seeds = max(1, int(getattr(
                            coevirag.global_config,
                            "rl_downstream_seed_top_k",
                            6,
                        )))
                        gate_advantage = (
                            delta
                            + downstream_positive_bonus * max(0.0, delta)
                            - seed_gate_cost * min(1.0, seed_count / max_gate_seeds)
                        )
                        start_node = record["start_node"]
                        seed_emb = coevirag.get_embedding_for_node(start_node)
                        query_emb = coevirag._get_query_embedding_tensor(query)
                        if seed_emb is not None:
                            gate_loss = agent.seed_gate_loss(
                                seed_emb.view(1, -1),
                                query_emb.view(1, -1),
                                keep_advantage=gate_advantage,
                                loss_weight=seed_gate_loss_weight,
                            )
                            if gate_loss is not None:
                                pending_losses.append(gate_loss)
                            with torch.no_grad():
                                keep_logit = agent.policy.seed_keep_logit(
                                    seed_emb.view(1, -1),
                                    query_emb.view(1, -1),
                                )
                                seed_gate_probabilities.append(
                                    float(torch.sigmoid(keep_logit).item())
                                )
                            seed_gate_advantages.append(gate_advantage)

                terminal_bonus = downstream_weight * credit
                total_epoch_reward += sum(trace["rewards"]) + terminal_bonus
                total_episodes += 1
                policy_loss = agent.consume_episode_trace(
                    trace,
                    terminal_bonus=terminal_bonus,
                )
                if policy_loss is not None:
                    pending_losses.append(policy_loss)
            if len(pending_losses) >= update_batch_episodes:
                agent.apply_policy_losses(pending_losses)
                pending_losses = []
        progress_bar.set_postfix({"Avg Reward": f"{total_epoch_reward / max(total_episodes, 1):.2f}"})

    agent.apply_policy_losses(pending_losses)
    avg_reward = total_epoch_reward / total_episodes if total_episodes else 0
    logger.info(f"Epoch {epoch_num} [Training] Completed. Avg Reward: {avg_reward:.2f}")
    if completed_start_skips:
        logger.info("Epoch %d [Fact Completed Starts] skipped=%d active_episodes=%d",
                    epoch_num, completed_start_skips, total_episodes)
    if downstream_enabled:
        positive_rate = float(np.mean(np.asarray(downstream_deltas) > 0.0)) if downstream_deltas else 0.0
        logger.info(
            "Epoch %d [Hierarchical Fact Reward] episodes=%d, "
            "baseline_utility=%.4f, seeded_utility=%.4f, delta=%.4f, "
            "positive_rate=%.4f, credit_mode=%s, mean_credit=%.4f, "
            "mean_abs_credit=%.4f, mean_fact_seeds=%.2f, "
            "recall_weights=%.2f/%.2f/%.2f",
            epoch_num,
            len(downstream_deltas),
            float(np.mean(downstream_baselines)) if downstream_baselines else 0.0,
            float(np.mean(downstream_seeded)) if downstream_seeded else 0.0,
            float(np.mean(downstream_deltas)) if downstream_deltas else 0.0,
            positive_rate,
            downstream_credit_mode,
            float(np.mean(downstream_credits)) if downstream_credits else 0.0,
            float(np.mean(np.abs(downstream_credits))) if downstream_credits else 0.0,
            float(np.mean(downstream_seed_counts)) if downstream_seed_counts else 0.0,
            downstream_recall_weights[0],
            downstream_recall_weights[1],
            downstream_recall_weights[2],
        )
        if seed_gate_enabled:
            logger.info(
                "Epoch %d [Fact Seed Gate] examples=%d, target_keep_rate=%.4f, "
                "mean_keep_advantage=%.4f, mean_keep_probability=%.4f, "
                "seed_cost=%.4f",
                epoch_num,
                len(seed_gate_advantages),
                float(np.mean(np.asarray(seed_gate_advantages) > 0.0))
                if seed_gate_advantages else 0.0,
                float(np.mean(seed_gate_advantages)) if seed_gate_advantages else 0.0,
                float(np.mean(seed_gate_probabilities))
                if seed_gate_probabilities else 0.0,
                seed_gate_cost,
            )
    return avg_reward


def get_paragraph_training_seeds(
        coevirag: CoEviRAG,
        query: str,
        top_k_facts: List[Tuple],
        top_k_fact_indices: List[int],
        query_fact_scores: np.ndarray,
        fact_passage_info: Optional[Dict[str, Dict]] = None) -> List[str]:
    fact_seed_scores = {}

    def update_fact_seed(passage_key: str, score: float) -> None:
        if passage_key not in coevirag.passage_key_to_idx:
            return
        fact_seed_scores[passage_key] = max(
            fact_seed_scores.get(passage_key, -float("inf")),
            float(score),
        )

    dpr_ids, dpr_scores = coevirag.dense_passage_retrieval(query)
    dpr_scores = min_max_normalize(dpr_scores)
    dpr_top_k = getattr(coevirag.global_config, "paragraph_train_dpr_top_k", 10)
    dpr_seed_scores = {}
    for doc_id, score in zip(dpr_ids[:dpr_top_k], dpr_scores[:dpr_top_k]):
        passage_key = coevirag.passage_node_keys[int(doc_id)]
        dpr_seed_scores[passage_key] = max(
            dpr_seed_scores.get(passage_key, -float("inf")),
            float(score),
        )

    if fact_passage_info is not None:
        max_fact_passages = getattr(
            coevirag.global_config,
            "paragraph_agent_seed_top_k",
            20,
        )
        for passage_key, info in sorted(
                fact_passage_info.items(),
                key=lambda item: (
                    item[1].get("score", -float("inf")),
                    item[0],
                ),
                reverse=True)[:max_fact_passages]:
            update_fact_seed(passage_key, float(info.get("score", 0.0)))
    else:
        graph_seeds = coevirag.get_ranked_seed_nodes(
            top_k_facts=top_k_facts,
            top_k_fact_indices=top_k_fact_indices,
            query_fact_scores=query_fact_scores,
            max_seeds=getattr(coevirag.global_config, "rl_seed_top_k", 12),
            include_fact_nodes=True,
        )
        for node_key, seed_score in graph_seeds:
            for passage_key in coevirag.get_supporting_passages_for_node(node_key):
                update_fact_seed(passage_key, float(seed_score))

    max_seeds = getattr(coevirag.global_config, "paragraph_train_seed_top_k", 3)
    if max_seeds <= 0:
        return []

    fact_ranked = sorted(
        fact_seed_scores.items(),
        key=lambda item: (item[1], item[0]),
        reverse=True,
    )
    if (
            fact_ranked
            and getattr(
                coevirag.global_config,
                "paragraph_fact_seed_score_mode",
                "fixed",
            ) == "normalized"):
        normalized_scores = min_max_normalize(
            np.asarray([score for _, score in fact_ranked], dtype=np.float32)
        )
        fact_ranked = [
            (key, float(score))
            for (key, _), score in zip(fact_ranked, normalized_scores)
        ]
    dpr_ranked = sorted(
        dpr_seed_scores.items(),
        key=lambda item: (item[1], item[0]),
        reverse=True,
    )

    seed_mode = getattr(coevirag.global_config, "paragraph_seed_mode", "hybrid")
    if seed_mode == "fact_only":
        return [key for key, _ in fact_ranked[:max_seeds]]
    if seed_mode == "dpr_only":
        return [key for key, _ in dpr_ranked[:max_seeds]]
    if seed_mode != "hybrid":
        raise ValueError(f"Unsupported paragraph seed mode: {seed_mode}")

    fact_ratio = min(
        1.0,
        max(
            0.0,
            float(
                getattr(
                    coevirag.global_config,
                    "paragraph_train_fact_ratio",
                    0.5,
                )
            ),
        ),
    )
    fact_quota = int(round(max_seeds * fact_ratio))
    if fact_ranked and dpr_ranked and max_seeds > 1:
        fact_quota = min(max_seeds - 1, max(1, fact_quota))
    dpr_quota = max_seeds - fact_quota

    selected = []
    seen = set()

    def add_ranked(ranked_seeds, limit):
        if limit <= 0:
            return
        added = 0
        for passage_key, _ in ranked_seeds:
            if passage_key in seen:
                continue
            selected.append(passage_key)
            seen.add(passage_key)
            added += 1
            if added >= limit:
                break

    add_ranked(fact_ranked, fact_quota)
    add_ranked(dpr_ranked, dpr_quota)
    if len(selected) < max_seeds:
        remaining = sorted(
            [
                (key, score, "fact")
                for key, score in fact_ranked
                if key not in seen
            ] + [
                (key, score, "dpr")
                for key, score in dpr_ranked
                if key not in seen
            ],
            key=lambda item: (item[1], item[0], item[2]),
            reverse=True,
        )
        add_ranked(
            [(key, score) for key, score, _ in remaining],
            max_seeds - len(selected),
        )
    return selected[:max_seeds]


def train_paragraph_agent_epoch(coevirag: CoEviRAG, training_data: List[Dict], epoch_num: int, gold_emb_cache: Dict[str, torch.Tensor]):
    if not getattr(coevirag.global_config, "train_paragraph_agent", True):
        logger.info(f"Epoch {epoch_num} [Paragraph Training] Skipped.")
        return 0.0

    agent = coevirag.paragraph_agent
    total_epoch_reward = 0
    total_episodes = 0
    completed_start_skips = 0
    pending_losses = []
    update_batch_episodes = max(
        1,
        int(getattr(coevirag.global_config, "rl_update_batch_episodes", 1)),
    )
    progress_bar = tqdm(training_data, desc=f"Epoch {epoch_num} [Paragraph Training]")
    coevirag.paragraph_policy_network.train()
    phase_seconds = {
        "fact_ranking": 0.0,
        "fact_seed_search": 0.0,
        "seed_fusion": 0.0,
        "paragraph_episodes": 0.0,
        "policy_updates": 0.0,
    }

    for data_point in progress_bar:
        query, gold_docs = data_point['query'], data_point['gold_docs']
        gold_doc_embs = None
        if gold_docs:
            embs_list = [gold_emb_cache[doc] for doc in gold_docs if doc in gold_emb_cache]
            if embs_list:
                gold_doc_embs = torch.stack(embs_list)

        phase_started = time.perf_counter()
        query_fact_scores = coevirag.get_fact_scores(query)
        top_k_fact_indices, top_k_facts = coevirag.rank_facts(query, query_fact_scores)
        phase_seconds["fact_ranking"] += time.perf_counter() - phase_started
        fact_passage_info = None
        fact_seed_source = getattr(
            coevirag.global_config,
            "paragraph_train_fact_seed_source",
            "linking",
        )
        if fact_seed_source == "policy":
            phase_started = time.perf_counter()
            fact_was_training = coevirag.rl_policy_network.training
            coevirag.rl_policy_network.eval()
            with torch.no_grad():
                fact_passage_info, _ = coevirag.run_rl_graph_search_rerank(
                    query=query,
                    top_k_facts=top_k_facts,
                    top_k_fact_indices=top_k_fact_indices,
                    query_fact_scores=query_fact_scores,
                    max_seeds=getattr(
                        coevirag.global_config,
                        "paragraph_train_fact_search_seed_top_k",
                        4,
                    ),
                    max_steps=getattr(
                        coevirag.global_config,
                        "paragraph_train_fact_search_steps",
                        3,
                    ),
                )
            if fact_was_training:
                coevirag.rl_policy_network.train()
            phase_seconds["fact_seed_search"] += time.perf_counter() - phase_started
        elif fact_seed_source != "linking":
            raise ValueError(
                f"Unsupported Paragraph Fact seed source: {fact_seed_source}"
            )
        phase_started = time.perf_counter()
        start_passages = get_paragraph_training_seeds(
            coevirag=coevirag,
            query=query,
            top_k_facts=top_k_facts,
            top_k_fact_indices=top_k_fact_indices,
            query_fact_scores=query_fact_scores,
            fact_passage_info=fact_passage_info,
        )
        phase_seconds["seed_fusion"] += time.perf_counter() - phase_started
        if not start_passages:
            continue

        max_steps = getattr(coevirag.global_config, "paragraph_agent_max_steps", 2)
        for start_passage in start_passages:
            phase_started = time.perf_counter()
            agent.run_paragraph_episode(
                coevirag_instance=coevirag,
                start_node=start_passage,
                query=query,
                gold_docs=gold_docs,
                gold_doc_embs=gold_doc_embs,
                max_steps=max_steps,
                reward_gold=coevirag.rl_reward_gold,
                reward_step=coevirag.rl_reward_step
            )
            phase_seconds["paragraph_episodes"] += time.perf_counter() - phase_started
            if getattr(agent, "last_episode_start_skipped", False):
                completed_start_skips += 1
                continue
            total_epoch_reward += sum(agent.rewards)
            if agent.rewards:
                total_episodes += 1
            phase_started = time.perf_counter()
            policy_loss = agent.consume_episode_loss()
            if policy_loss is not None:
                pending_losses.append(policy_loss)
            if len(pending_losses) >= update_batch_episodes:
                agent.apply_policy_losses(pending_losses)
                pending_losses = []
            phase_seconds["policy_updates"] += time.perf_counter() - phase_started

        progress_bar.set_postfix({"Avg Reward": f"{total_epoch_reward / max(total_episodes, 1):.2f}"})

    phase_started = time.perf_counter()
    agent.apply_policy_losses(pending_losses)
    phase_seconds["policy_updates"] += time.perf_counter() - phase_started
    avg_reward = total_epoch_reward / total_episodes if total_episodes else 0
    logger.info(f"Epoch {epoch_num} [Paragraph Training] Completed. Avg Reward: {avg_reward:.2f}")
    if completed_start_skips:
        logger.info("Epoch %d [Paragraph Completed Starts] skipped=%d active_episodes=%d",
                    epoch_num, completed_start_skips, total_episodes)
    measured_total = sum(phase_seconds.values())
    logger.info(
        "Epoch %d [Paragraph Timing] total=%.2fs fact_ranking=%.2fs "
        "fact_seed_search=%.2fs seed_fusion=%.2fs paragraph_episodes=%.2fs "
        "policy_updates=%.2fs",
        epoch_num,
        measured_total,
        phase_seconds["fact_ranking"],
        phase_seconds["fact_seed_search"],
        phase_seconds["seed_fusion"],
        phase_seconds["paragraph_episodes"],
        phase_seconds["policy_updates"],
    )
    return avg_reward


def train_learned_coordinator(
        coevirag: CoEviRAG,
        train_queries: List[str],
        train_gold_docs: List[List[str]],
        max_queries: int,
        max_negatives_per_query: int,
        epochs: int,
        lr: float,
        loss_type: str,
        model_type: str = "linear",
        hidden_dim: int = 16,
        feature_cache_path: str = ""):
    logger.info("--- Training learned coordinator gate ---")
    loss_type = (loss_type or "bce").lower()
    x_np = y_np = qid_np = None
    cache_path = Path(feature_cache_path) if feature_cache_path else None
    if cache_path and cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        cached_x = cached["features"]
        if cached_x.ndim == 2 and cached_x.shape[1] == 19:
            x_np = cached_x.astype(np.float32)
            y_np = cached["labels"].astype(np.float32)
            qid_np = cached["query_ids"].astype(np.int64)
            logger.info(f"Loaded coordinator feature cache: {cache_path} ({len(y_np)} examples)")
        else:
            logger.warning(f"Ignoring incompatible coordinator feature cache: {cache_path}")

    if x_np is None:
        x_np, y_np, qid_np = coevirag.collect_coordinator_examples(
            queries=train_queries,
            gold_docs=train_gold_docs,
            max_queries=max_queries,
            max_negatives_per_query=max_negatives_per_query,
            return_query_ids=True,
        )
        if cache_path and len(y_np) > 0:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_path,
                features=x_np,
                labels=y_np,
                query_ids=qid_np,
            )
            logger.info(f"Saved coordinator feature cache: {cache_path}")
    if len(y_np) == 0 or y_np.sum() == 0:
        logger.warning("No positive coordinator examples found; learned coordinator disabled for this run.")
        coevirag.use_learned_coordinator = False
        return

    mean = x_np.mean(axis=0)
    raw_std = x_np.std(axis=0)
    active_features = raw_std > 1e-6
    std = np.where(active_features, raw_std + 1e-6, 1.0)
    x = torch.from_numpy((x_np - mean) / std).float().to(coevirag.device)
    y = torch.from_numpy(y_np).float().view(-1, 1).to(coevirag.device)
    qids = torch.from_numpy(qid_np).long().to(coevirag.device) if qid_np is not None else None
    rank_weights_np = np.ones(len(y_np), dtype=np.float32)
    for query_id in np.unique(qid_np):
        query_indices = np.flatnonzero(qid_np == query_id)
        ranked_indices = query_indices[np.argsort(-x_np[query_indices, 11])]
        for rank, example_idx in enumerate(ranked_indices, start=1):
            rank_weights_np[example_idx] = 3.0 if rank <= 2 else (2.0 if rank <= 5 else (1.5 if rank <= 10 else 1.0))
    rank_weights = torch.from_numpy(rank_weights_np).float().to(coevirag.device)

    if model_type == "mlp":
        model = torch.nn.Sequential(
            torch.nn.Linear(x.shape[1], max(2, hidden_dim)),
            torch.nn.ReLU(),
            torch.nn.Linear(max(2, hidden_dim), 1),
        ).to(coevirag.device)
    else:
        model = torch.nn.Linear(x.shape[1], 1).to(coevirag.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    pos = float(y_np.sum())
    neg = float(len(y_np) - y_np.sum())
    pos_weight = torch.tensor([max(1.0, neg / max(pos, 1.0))], device=coevirag.device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for _ in range(max(1, epochs)):
        optimizer.zero_grad()
        logits = model(x)
        if loss_type in {"query_pairwise", "rank_weighted_pairwise"}:
            query_losses = []
            for qid in torch.unique(qids):
                mask = qids == qid
                q_logits = logits[mask].view(-1)
                q_labels = y[mask].view(-1)
                pos_logits = q_logits[q_labels > 0.5]
                neg_logits = q_logits[q_labels <= 0.5]
                if len(pos_logits) > 0 and len(neg_logits) > 0:
                    pair_diffs = pos_logits.view(-1, 1) - neg_logits.view(1, -1)
                    pair_losses = torch.nn.functional.softplus(-pair_diffs)
                    if loss_type == "rank_weighted_pairwise":
                        neg_weights = rank_weights[mask][q_labels <= 0.5].view(1, -1)
                        weighted_loss = (pair_losses * neg_weights).sum() / (
                            neg_weights.sum() * max(1, len(pos_logits))
                        )
                        query_losses.append(weighted_loss)
                    else:
                        query_losses.append(pair_losses.mean())
            if query_losses:
                loss = torch.stack(query_losses).mean()
            else:
                loss = loss_fn(logits, y)
        elif loss_type == "pairwise":
            pos_logits = logits[y.view(-1) > 0.5].view(-1)
            neg_logits = logits[y.view(-1) <= 0.5].view(-1)
            if len(pos_logits) > 0 and len(neg_logits) > 0:
                pair_diffs = pos_logits.view(-1, 1) - neg_logits.view(1, -1)
                loss = torch.nn.functional.softplus(-pair_diffs).mean()
            else:
                loss = loss_fn(logits, y)
        else:
            loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = model(x)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        acc = (preds.eq(y)).float().mean().item()

    if model_type == "mlp":
        hidden_layer = model[0]
        output_layer = model[2]
        hidden_weight = hidden_layer.weight.detach().cpu().numpy().astype(np.float32)
        hidden_weight[:, ~active_features] = 0.0
        coevirag.coordinator_mlp_state = {
            "hidden_weight": hidden_weight,
            "hidden_bias": hidden_layer.bias.detach().cpu().numpy().astype(np.float32),
            "output_weight": output_layer.weight.detach().cpu().numpy().reshape(-1).astype(np.float32),
            "output_bias": float(output_layer.bias.detach().cpu().numpy().reshape(-1)[0]),
        }
        coevirag.coordinator_weights = None
        coevirag.coordinator_bias = 0.0
    else:
        coordinator_weights = model.weight.detach().cpu().numpy().reshape(-1)
        coordinator_weights[~active_features] = 0.0
        coevirag.coordinator_weights = coordinator_weights
        coevirag.coordinator_bias = float(model.bias.detach().cpu().numpy().reshape(-1)[0])
        coevirag.coordinator_mlp_state = None
    coevirag.coordinator_feature_mean = mean.astype(np.float32)
    coevirag.coordinator_feature_std = std.astype(np.float32)
    coevirag.use_learned_coordinator = True
    logger.info(
        f"Learned coordinator trained. examples={len(y_np)}, positives={int(pos)}, "
        f"model_type={model_type}, loss_type={loss_type}, "
        f"loss={loss.item():.4f}, train_acc={acc:.4f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Train CoEviRAG RL Agent")
    parser.add_argument('--dataset', type=str, default='2wikimultihopqa', help='Dataset name for evaluation')
    parser.add_argument('--llm_base_url', type=str, default='https://integrate.api.nvidia.com/v1', help='LLM base URL')
    parser.add_argument('--llm_name', type=str, default='openai/gpt-oss-120b', help='LLM name')
    parser.add_argument('--llm_reasoning_effort', choices=('none', 'low', 'medium', 'high'), default='low', help='Reasoning effort for supported APIs')
    parser.add_argument('--max_new_tokens', type=int, default=2048, help='Maximum completion tokens for OpenIE and LLM calls')
    parser.add_argument('--openie_batch_size', type=int, default=200, help='OpenIE passages saved per durable checkpoint')
    parser.add_argument('--openie_num_workers', type=int, default=16, help='Concurrent OpenIE API workers')
    parser.add_argument('--openie_pipeline_mode', choices=('staged', 'per_document'), default='staged', help='Schedule OpenIE as two global stages or one two-stage pipeline per passage')
    parser.add_argument('--openie_max_new_passages', type=int, default=0, help='Process at most this many pending OpenIE passages; 0 is unlimited')
    parser.add_argument('--openie_only', type=str, default='false', help='Exit after OpenIE/index processing without retrieval setup')
    parser.add_argument('--openie_results_path', type=str, default='', help='Explicit OpenIE cache path, independent of the query/QA LLM name')
    parser.add_argument('--openie_empty_retry_attempts', type=int, default=1, help='Corrective retries for empty OpenIE JSON')
    parser.add_argument('--openie_batch_retry_attempts', type=int, default=3, help='Retry an OpenIE batch before stopping without a checkpoint')
    parser.add_argument('--llm_max_retry_attempts', type=int, default=3, help='Retries for individual LLM API requests')
    parser.add_argument('--llm_request_timeout', type=float, default=30.0, help='Timeout in seconds for each LLM API request')
    parser.add_argument('--query_extraction_max_workers', type=int, default=4, help='Concurrent workers for query OpenIE extraction')
    parser.add_argument('--embedding_name', type=str, default='Qwen/Qwen3-Embedding-4B', help='Embedding model name')
    parser.add_argument('--graph_variant', type=str, default='', help='Optional isolated graph artifact suffix')
    parser.add_argument('--synonymy_edge_topk', type=int, default=2047, help='KNN candidates used to construct synonym edges')
    parser.add_argument('--synonymy_edge_sim_threshold', type=float, default=0.8, help='Minimum synonym edge similarity')
    parser.add_argument('--synonymy_edge_max_neighbors', type=int, default=100, help='Maximum synonym neighbors per entity')
    parser.add_argument('--synonymy_edge_mutual_knn', type=str, default='false', help='Keep only mutual synonym neighbors')
    parser.add_argument('--force_index_from_scratch', type=str, default='false', help='Force re-indexing')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs for RL training')
    parser.add_argument('--save_dir', type=str, default='outputs', help='Save directory for index, models, and data splits')
    parser.add_argument('--policy_run_dir', type=str, default='', help='Optional isolated directory for loading and saving policy checkpoints')
    parser.add_argument('--eval_split', type=str, default='val', choices=['val', 'test'], help='Dataset split used for evaluation')
    parser.add_argument('--val_split', type=float, default=0.1, help='Fraction of data to use for validation (e.g., 0.1 for 10%%)')
    parser.add_argument('--test_split', type=float, default=0.5, help='Fraction of data to use for the final test set')
    parser.add_argument('--validate_every_n_epochs', type=int, default=5, help='Run validation every N epochs')
    parser.add_argument('--early_stopping_patience', type=int, default=0, help='Stop after this many validation checks without improvement; 0 disables early stopping')
    parser.add_argument('--early_stopping_min_delta', type=float, default=0.0, help='Minimum checkpoint-score increase counted as an improvement')
    parser.add_argument('--train_limit', type=int, default=0, help='Use only the first N training samples when > 0')
    parser.add_argument('--val_offset', type=int, default=0, help='Skip the first N validation samples before applying val_limit')
    parser.add_argument('--val_limit', type=int, default=0, help='Use only the first N validation samples when > 0')
    parser.add_argument('--eval_on_train', type=str, default='false', help='Use the selected training samples as the evaluation split for offline gate-data experiments')
    parser.add_argument('--use_ppr_rerank', type=str, default='false', help='Whether to include PPR in final reranking')
    parser.add_argument('--w_ppr', type=float, default=0.0, help='PPR score weight in final reranking')
    parser.add_argument('--w_policy', type=float, default=0.2, help='Graph/RL path-score weight in final reranking')
    parser.add_argument('--normalize_rerank_features', type=str, default='false', help='Normalize reranking features within each query')
    parser.add_argument('--normalize_path_by_length', type=str, default='false', help='Use mean per-step path scores before normalization')
    parser.add_argument('--rerank_fusion_mode', type=str, default='score', choices=['score', 'similarity', 'rank_rrf', 'query_conditioned'], help='Fuse raw scores, pure similarity, source-local ranks, or use the unified learned ranker')
    parser.add_argument('--rank_fusion_rrf_k', type=int, default=60, help='RRF smoothing constant for rank-based candidate fusion')
    parser.add_argument('--fact_head_protect_k', type=int, default=0, help='Reserve first k final ranks for Fact-agent candidates')
    parser.add_argument('--similarity_head_protect_k', type=int, default=0, help='Reserve first k final ranks for dense-similarity candidates')
    parser.add_argument('--query_conditioned_ranker_path', type=str, default='', help='Checkpoint for unified query-conditioned candidate reranking')
    parser.add_argument('--query_conditioned_fact_budget', type=int, default=200, help='Fact candidates passed to the unified ranker')
    parser.add_argument('--query_conditioned_paragraph_budget', type=int, default=200, help='Paragraph candidates passed to the unified ranker')
    parser.add_argument('--query_conditioned_repair_budget', type=int, default=50, help='Verified Repair candidates passed to the unified ranker')
    parser.add_argument('--query_conditioned_similarity_budget', type=int, default=0, help='Global dense-similarity candidates added as a safety channel')
    parser.add_argument('--set_aware_selector_path', type=str, default='', help='Optional semantic evidence-set selector applied after unified ranking')
    parser.add_argument('--set_aware_selection_mode', type=str, default='select2', choices=['select2', 'dynamic_setmax', 'dynamic_setmax_recovery'], help='Evidence-set selection policy')
    parser.add_argument('--set_aware_anchor_mode', type=str, default='unary', choices=['unary', 'dpr'], help='Use unified-ranker or DPR scores for the evidence anchor and selector prior')
    parser.add_argument('--retrieval_source_mode', type=str, default='dual', choices=['dual', 'fact_only', 'paragraph_only'], help='Select candidate sources for isolation experiments')
    parser.add_argument('--source_diagnostics', type=str, default='false', help='Report per-source Recall and complete-evidence rates')
    parser.add_argument('--query_fact_score_mode', type=str, default='concat', choices=['concat', 'slot_max', 'slot_mean', 'slot_hybrid'], help='Aggregate structured query triples jointly, per relation slot, or by hybrid scoring')
    parser.add_argument('--query_slot_blend_alpha', type=float, default=0.5, help='Slot-max weight used by slot_hybrid query scoring')
    parser.add_argument('--w_query', type=float, default=0.5, help='Raw-query weight versus structured-query fact scoring')
    parser.add_argument('--cache_static_retrieval', type=str, default='true', help='Cache static Fact/DPR scores and rankings across training epochs')
    parser.add_argument('--precompute_paragraph_knn', type=str, default='false', help='Precompute paragraph semantic Top-K neighbors in GPU batches')
    parser.add_argument('--paragraph_knn_batch_size', type=int, default=256, help='GPU query batch size for paragraph KNN precomputation')
    parser.add_argument('--batch_fact_beam_scoring', type=str, default='true', help='Score active Fact beams in one policy-network forward per step')
    parser.add_argument('--fact_traversal_mode', type=str, default='undirected', choices=['undirected', 'hard_directed', 'role_aware'], help='Fact-KG traversal direction for training and inference')
    parser.add_argument('--w_fact_policy', type=float, default=0.15, help='Normalized Fact-agent score weight')
    parser.add_argument('--w_paragraph_policy', type=float, default=0.15, help='Normalized Paragraph-agent score weight')
    parser.add_argument('--w_sim', type=float, default=0.6, help='Query-passage similarity weight in final reranking')
    parser.add_argument('--w_repair_confidence', type=float, default=0.1, help='Verified repair confidence weight')
    parser.add_argument('--w_source_agreement', type=float, default=0.05, help='Bonus for passages found by multiple agents')
    parser.add_argument('--use_learned_coordinator', type=str, default='false', help='Whether to train/use a lightweight learned coordinator gate')
    parser.add_argument('--coordinator_train_limit', type=int, default=0, help='Use first N train queries for coordinator training; 0 means all current training data')
    parser.add_argument('--coordinator_max_negatives', type=int, default=30, help='Max negative candidates per query for coordinator training')
    parser.add_argument('--coordinator_epochs', type=int, default=80, help='Epochs for lightweight coordinator training')
    parser.add_argument('--coordinator_lr', type=float, default=0.05, help='Learning rate for lightweight coordinator training')
    parser.add_argument('--coordinator_loss', type=str, default='bce', choices=['bce', 'pairwise', 'query_pairwise', 'rank_weighted_pairwise'], help='Loss for lightweight coordinator training')
    parser.add_argument('--coordinator_model_type', type=str, default='linear', choices=['linear', 'mlp'], help='Lightweight coordinator model family')
    parser.add_argument('--coordinator_hidden_dim', type=int, default=16, help='Hidden dimension for the MLP coordinator')
    parser.add_argument('--coordinator_feature_cache', type=str, default='', help='Optional NPZ cache for frozen-agent coordinator training features')
    parser.add_argument('--coordinator_blend_alpha', type=float, default=0.1, help='Residual blend weight for learned coordinator logits')
    parser.add_argument('--coordinator_protect_top_k', type=int, default=0, help='Keep top-k heuristic candidates unchanged before applying learned coordinator residual')
    parser.add_argument('--coordinator_margin_gate_top_k', type=int, default=0, help='Disable coordinator when the heuristic top-k boundary is not residual-safe')
    parser.add_argument('--coordinator_margin_gate_threshold', type=float, default=0.0, help='Safe top-k margin; <=0 uses coordinator_blend_alpha')
    parser.add_argument('--rl_learning_rate', type=float, default=1e-4, help='Learning rate for Fact/Paragraph policy optimization')
    parser.add_argument('--rl_reward_mode', type=str, default='legacy', choices=['legacy', 'coverage_progress'], help='RL reward shaping mode')
    parser.add_argument('--rl_skip_completed_starts', type=str, default='false', help='Skip action updates for coverage_progress starts already covering all gold; disabled for hierarchical rewards')
    parser.add_argument('--rl_evidence_rank_loss_weight', type=float, default=0.0, help='Optional local-evidence action ranking loss; zero preserves RL-only training')
    parser.add_argument('--rl_progress_scale', type=float, default=2.0, help='Signed gold-similarity progress reward scale')
    parser.add_argument('--rl_terminal_progress_scale', type=float, default=0.25, help='Terminal reward scale for improvement over the starting node')
    parser.add_argument('--rl_missing_gold_penalty', type=float, default=0.25, help='Terminal penalty scale for uncovered gold passages')
    parser.add_argument('--rl_complete_evidence_bonus', type=float, default=0.0, help='Terminal bonus for covering every gold passage in one episode')
    parser.add_argument('--rl_advantage_mode', type=str, default='episode_normalized', choices=['episode_normalized', 'ema_baseline', 'raw'], help='REINFORCE advantage estimator')
    parser.add_argument('--rl_baseline_momentum', type=float, default=0.95, help='EMA momentum used by the RL return baseline')
    parser.add_argument('--rl_entropy_beta', type=float, default=0.0, help='Entropy regularization weight for RL policy updates')
    parser.add_argument('--rl_grad_clip', type=float, default=0.0, help='Gradient clipping norm for RL policy updates; <=0 disables clipping')
    parser.add_argument('--rl_update_batch_episodes', type=int, default=1, help='Number of complete episodes combined into one optimizer step')
    parser.add_argument('--rl_seed_top_k', type=int, default=12, help='Number of inference seeds for RL graph search')
    parser.add_argument('--rl_train_seed_top_k', type=int, default=3, help='Number of training seeds per query')
    parser.add_argument('--rl_hierarchical_downstream_reward', type=str, default='false', help='Reward Fact episodes by frozen Paragraph improvement over DPR-only retrieval')
    parser.add_argument('--rl_downstream_reward_weight', type=float, default=1.0, help='Scale of the Paragraph downstream-gain terminal reward')
    parser.add_argument('--rl_downstream_complete_weight', type=float, default=0.25, help='Complete-evidence weight inside downstream Paragraph utility')
    parser.add_argument('--rl_downstream_recall2_weight', type=float, default=1.0, help='Recall@2 weight inside downstream Paragraph utility')
    parser.add_argument('--rl_downstream_recall5_weight', type=float, default=1.0, help='Recall@5 weight inside downstream Paragraph utility')
    parser.add_argument('--rl_downstream_recall10_weight', type=float, default=1.0, help='Recall@10 weight inside downstream Paragraph utility')
    parser.add_argument('--rl_downstream_reward_clip', type=float, default=1.0, help='Absolute clipping bound for per-episode downstream utility gain')
    parser.add_argument('--rl_downstream_seed_top_k', type=int, default=6, help='Maximum Fact-path paragraph seeds used to estimate downstream gain')
    parser.add_argument('--rl_downstream_credit_mode', type=str, default='raw', choices=['raw', 'group_relative'], help='Use raw downstream gain or within-query counterfactual credit')
    parser.add_argument('--rl_downstream_positive_bonus', type=float, default=0.0, help='Extra reward scale for Fact paths with positive Paragraph gain')
    parser.add_argument('--rl_paragraph_seed_gate_enabled', type=str, default='false', help='Learn and apply a Fact-seed KEEP/SKIP gate before Paragraph search')
    parser.add_argument('--rl_paragraph_seed_gate_threshold', type=float, default=0.5, help='Inference KEEP probability threshold for Fact-to-Paragraph seeds')
    parser.add_argument('--rl_paragraph_seed_gate_min_keep', type=int, default=0, help='Minimum highest-probability Fact seeds retained per query after gating')
    parser.add_argument('--rl_seed_gate_cost', type=float, default=0.01, help='Counterfactual cost paid when a Fact seed is passed to Paragraph')
    parser.add_argument('--rl_seed_gate_loss_weight', type=float, default=5.0, help='Optimization weight for the counterfactual KEEP/SKIP loss')
    parser.add_argument('--train_fact_agent', type=str, default='true', help='Whether to train the fact navigation agent')
    parser.add_argument('--use_paragraph_agent', type=str, default='true', help='Whether to use paragraph agent during retrieval')
    parser.add_argument('--train_paragraph_agent', type=str, default='true', help='Whether to train paragraph agent separately')
    parser.add_argument('--use_evidence_paragraph_graph', type=str, default='false', help='Whether paragraph graph uses shared fact/entity evidence edges')
    parser.add_argument('--evidence_paragraph_max_candidates', type=int, default=80, help='Max evidence-aware paragraph candidates per node')
    parser.add_argument('--evidence_paragraph_max_entity_support', type=int, default=80, help='Skip shared-entity edges for entities with too many supporting passages')
    parser.add_argument('--paragraph_train_seed_top_k', type=int, default=3, help='Number of paragraph start passages per query')
    parser.add_argument('--paragraph_train_dpr_top_k', type=int, default=10, help='DPR top-k candidates for paragraph training seeds')
    parser.add_argument('--paragraph_train_fact_ratio', type=float, default=0.5, help='Fraction of Paragraph training starts reserved for Fact-mapped seeds')
    parser.add_argument('--paragraph_train_fact_seed_source', type=str, default='linking', choices=['linking', 'policy'], help='Use static linking or current Fact policy outputs for Paragraph training seeds')
    parser.add_argument('--paragraph_train_fact_search_seed_top_k', type=int, default=4, help='Fact policy seed count used only to generate Paragraph training starts')
    parser.add_argument('--paragraph_train_fact_search_steps', type=int, default=3, help='Fact policy search steps used only to generate Paragraph training starts')
    parser.add_argument('--paragraph_seed_mode', type=str, default='hybrid', choices=['hybrid', 'dpr_only', 'fact_only'], help='Paragraph inference seed coordination mode')
    parser.add_argument('--paragraph_fact_seed_score_mode', type=str, default='fixed', choices=['fixed', 'normalized'], help='Fact-mapped Paragraph seed scoring')
    parser.add_argument('--repair_enabled', type=str, choices=('false',), default='false', help=argparse.SUPPRESS)
    parser.add_argument('--repair_apply_results', type=str, choices=('false',), default='false', help=argparse.SUPPRESS)
    parser.add_argument('--fact_edge_dropout_rate', type=float, default=0.0, help='Deterministic in-memory entity-fact edge dropout rate')
    parser.add_argument('--fact_edge_dropout_seed', type=int, default=42, help='Seed for deterministic entity-fact edge dropout')
    parser.add_argument('--fact_node_dropout_rate', type=float, default=0.0, help='Deterministic in-memory complete OpenIE fact-node dropout rate')
    parser.add_argument('--fact_node_dropout_seed', type=int, default=42, help='Seed for deterministic complete fact-node dropout')
    parser.add_argument('--targeted_bridge_fact_dropout', type=str, default='false', help='Remove gold-document bridge fact nodes to simulate missing OpenIE triples')
    parser.add_argument('--targeted_bridge_passage_mask', type=str, default='false', help='Hide bridge-supporting gold passages from Fact/Paragraph agents in controlled sparse evaluation')
    parser.add_argument('--targeted_bridge_facts_per_query', type=int, default=1, help='Bridge fact nodes removed per evaluation query')
    parser.add_argument('--query_relation_fact_dropout', type=str, default='false', help='Remove query-entity/relation-matched facts without using gold annotations')
    parser.add_argument('--query_relation_passage_mask', type=str, default='false', help='Hide source passages of query-relation-matched facts from base retrieval')
    parser.add_argument('--query_relation_facts_per_query', type=int, default=1, help='Query-relation-matched fact nodes removed per evaluation query')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducible RL experiments')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Torch device used by retrieval policy networks')
    parser.add_argument('--save_policy', type=str, default='true', help='Whether to save best/latest policy checkpoints')
    parser.add_argument('--save_training_state', type=str, default='false', help='Save latest policies, optimizers, EMA baselines, and RNG states after every epoch')
    parser.add_argument('--training_state_path', type=str, default='', help='Path for the latest resumable training state; defaults to the policy directory')
    parser.add_argument('--resume_training_state', type=str, default='', help='Resume from a complete training-state checkpoint')
    parser.add_argument('--policy_snapshot_epochs', type=str, default='', help='Comma-separated epochs or all; save both policies for final-evidence checkpoint selection')
    parser.add_argument('--checkpoint_metric', type=str, default='recall10', choices=['recall10', 'mean_recall_2_5_10'], help='Validation metric used to select the best policy checkpoint')
    parser.add_argument('--validate_before_training', type=str, default='false', help='Evaluate epoch-0 policies before training so a degraded epoch cannot replace the baseline')
    parser.add_argument('--retrieval_output_json', type=str, default='', help='Optional path for per-query ranked validation documents and scores')
    parser.set_defaults(repair_enabled="false", repair_apply_results="false",
                        graph_variant="unified_v1", fact_traversal_mode="role_aware")
    args = parser.parse_args()
    from coevirag.release_protocol import validate_navigation_args
    validate_navigation_args(args)
    if not math.isfinite(args.rl_evidence_rank_loss_weight) or args.rl_evidence_rank_loss_weight < 0:
        parser.error("--rl_evidence_rank_loss_weight must be finite and nonnegative")
    if args.rl_evidence_rank_loss_weight > 0 and args.rl_reward_mode != "coverage_progress":
        parser.error("Evidence action ranking supervision requires coverage_progress")

    logger.info("Experiment arguments: %s", vars(args))

    set_random_seed(args.random_seed)

    dataset_name = args.dataset
    save_dir = os.path.join(args.save_dir, dataset_name, "cache", "index")

    # --- 1. Prepare Datasets ---
    dataset_output_dir = f"datasets/{dataset_name}"
    train_samples, val_samples = prepare_datasets(
        dataset_name=dataset_name,
        base_dir=dataset_output_dir,
        val_split=args.val_split,
        test_split=args.test_split
    )
    if args.eval_split == 'test':
        test_path = os.path.join(dataset_output_dir, "test.json")
        with open(test_path, "r") as f:
            val_samples = json.load(f)
        logger.info("Using test split for evaluation (%d samples).", len(val_samples))
    if args.train_limit and args.train_limit > 0:
        train_samples = train_samples[:args.train_limit]
    if args.val_offset and args.val_offset > 0:
        val_samples = val_samples[args.val_offset:]
    if args.val_limit and args.val_limit > 0:
        val_samples = val_samples[:args.val_limit]
    if string_to_bool(args.eval_on_train):
        eval_limit = args.val_limit if args.val_limit and args.val_limit > 0 else len(train_samples)
        val_samples = train_samples[:eval_limit]
        logger.info(
            "Offline gate-data mode: evaluating on %d selected training samples.",
            len(val_samples),
        )

    # Load Corpus
    corpus_path = f"datasets/{dataset_name}/{dataset_name}_corpus.json"
    
    if not os.path.exists(corpus_path):
        logger.error(f"Corpus file not found: {corpus_path}")
        return

    with open(corpus_path, "r") as f:
        corpus = json.load(f)
    docs = [f"{doc['title']}\n{doc['text']}" for doc in corpus]
    logger.info(f"Corpus loaded, total {len(docs)} documents.")

    # Prepare Data Structures for Training/Validation
    train_queries = [s['question'] for s in train_samples]
    train_gold_docs = get_gold_docs(train_samples, dataset_name)
    training_data = [{'query': q, 'gold_docs': gd} for q, gd in zip(train_queries, train_gold_docs)]
    
    val_queries = [s['question'] for s in val_samples]
    val_gold_docs = get_gold_docs(val_samples, dataset_name)

    # --- 2. Initialize CoEviRAG and Index ---
    
    config = BaseConfig(
        save_dir=save_dir,
        llm_base_url=args.llm_base_url,
        llm_name=args.llm_name,
        llm_reasoning_effort=args.llm_reasoning_effort,
        max_new_tokens=args.max_new_tokens,
        openie_batch_size=args.openie_batch_size,
        openie_num_workers=args.openie_num_workers,
        openie_pipeline_mode=args.openie_pipeline_mode,
        openie_results_path=args.openie_results_path,
        openie_max_new_passages=args.openie_max_new_passages,
        openie_empty_retry_attempts=args.openie_empty_retry_attempts,
        openie_batch_retry_attempts=args.openie_batch_retry_attempts,
        max_retry_attempts=args.llm_max_retry_attempts,
        llm_request_timeout=args.llm_request_timeout,
        query_extraction_max_workers=args.query_extraction_max_workers,
        dataset=dataset_name,
        embedding_model_name=args.embedding_name,
        graph_variant=args.graph_variant,
        synonymy_edge_topk=args.synonymy_edge_topk,
        synonymy_edge_sim_threshold=args.synonymy_edge_sim_threshold,
        synonymy_edge_max_neighbors=args.synonymy_edge_max_neighbors,
        synonymy_edge_mutual_knn=string_to_bool(args.synonymy_edge_mutual_knn),
        seed=args.random_seed,
        execution_device=args.device,
        force_index_from_scratch=string_to_bool(args.force_index_from_scratch),
        force_openie_from_scratch=False, # Re-use OpenIE results during training usually
        retrieval_top_k=200,
        linking_top_k=10,
        qa_top_k=10,
        embedding_batch_size=8,
        openie_mode='online',
        use_ppr_rerank=string_to_bool(args.use_ppr_rerank),
        w_ppr=args.w_ppr,
        w_policy=args.w_policy,
        normalize_rerank_features=string_to_bool(args.normalize_rerank_features),
        normalize_path_by_length=string_to_bool(args.normalize_path_by_length),
        rerank_fusion_mode=args.rerank_fusion_mode,
        rank_fusion_rrf_k=args.rank_fusion_rrf_k,
        fact_head_protect_k=args.fact_head_protect_k,
        similarity_head_protect_k=args.similarity_head_protect_k,
        query_conditioned_ranker_path=args.query_conditioned_ranker_path,
        query_conditioned_fact_budget=args.query_conditioned_fact_budget,
        query_conditioned_paragraph_budget=args.query_conditioned_paragraph_budget,
        query_conditioned_repair_budget=args.query_conditioned_repair_budget,
        query_conditioned_similarity_budget=args.query_conditioned_similarity_budget,
        set_aware_selector_path=args.set_aware_selector_path,
        set_aware_selection_mode=args.set_aware_selection_mode,
        set_aware_anchor_mode=args.set_aware_anchor_mode,
        retrieval_source_mode=args.retrieval_source_mode,
        source_diagnostics=string_to_bool(args.source_diagnostics),
        query_fact_score_mode=args.query_fact_score_mode,
        query_slot_blend_alpha=args.query_slot_blend_alpha,
        w_query=args.w_query,
        cache_static_retrieval=string_to_bool(args.cache_static_retrieval),
        precompute_paragraph_knn=string_to_bool(args.precompute_paragraph_knn),
        paragraph_knn_batch_size=args.paragraph_knn_batch_size,
        batch_fact_beam_scoring=string_to_bool(args.batch_fact_beam_scoring),
        fact_traversal_mode=args.fact_traversal_mode,
        w_fact_policy=args.w_fact_policy,
        w_paragraph_policy=args.w_paragraph_policy,
        w_sim=args.w_sim,
        w_repair_confidence=args.w_repair_confidence,
        w_source_agreement=args.w_source_agreement,
        use_learned_coordinator=string_to_bool(args.use_learned_coordinator),
        coordinator_blend_alpha=args.coordinator_blend_alpha,
        coordinator_protect_top_k=args.coordinator_protect_top_k,
        coordinator_margin_gate_top_k=args.coordinator_margin_gate_top_k,
        coordinator_margin_gate_threshold=args.coordinator_margin_gate_threshold,
        rl_learning_rate=args.rl_learning_rate,
        rl_reward_mode=args.rl_reward_mode,
        rl_skip_completed_starts=string_to_bool(args.rl_skip_completed_starts),
        rl_evidence_rank_loss_weight=args.rl_evidence_rank_loss_weight,
        rl_progress_scale=args.rl_progress_scale,
        rl_terminal_progress_scale=args.rl_terminal_progress_scale,
        rl_missing_gold_penalty=args.rl_missing_gold_penalty,
        rl_complete_evidence_bonus=args.rl_complete_evidence_bonus,
        rl_advantage_mode=args.rl_advantage_mode,
        rl_baseline_momentum=args.rl_baseline_momentum,
        rl_entropy_beta=args.rl_entropy_beta,
        rl_grad_clip=args.rl_grad_clip,
        rl_update_batch_episodes=args.rl_update_batch_episodes,
        rl_seed_top_k=args.rl_seed_top_k,
        rl_train_seed_top_k=args.rl_train_seed_top_k,
        rl_hierarchical_downstream_reward=string_to_bool(args.rl_hierarchical_downstream_reward),
        rl_downstream_reward_weight=args.rl_downstream_reward_weight,
        rl_downstream_complete_weight=args.rl_downstream_complete_weight,
        rl_downstream_recall2_weight=args.rl_downstream_recall2_weight,
        rl_downstream_recall5_weight=args.rl_downstream_recall5_weight,
        rl_downstream_recall10_weight=args.rl_downstream_recall10_weight,
        rl_downstream_reward_clip=args.rl_downstream_reward_clip,
        rl_downstream_seed_top_k=args.rl_downstream_seed_top_k,
        rl_downstream_credit_mode=args.rl_downstream_credit_mode,
        rl_downstream_positive_bonus=args.rl_downstream_positive_bonus,
        rl_paragraph_seed_gate_enabled=string_to_bool(args.rl_paragraph_seed_gate_enabled),
        rl_paragraph_seed_gate_threshold=args.rl_paragraph_seed_gate_threshold,
        rl_paragraph_seed_gate_min_keep=args.rl_paragraph_seed_gate_min_keep,
        rl_seed_gate_cost=args.rl_seed_gate_cost,
        rl_seed_gate_loss_weight=args.rl_seed_gate_loss_weight,
        use_paragraph_agent=string_to_bool(args.use_paragraph_agent),
        train_paragraph_agent=string_to_bool(args.train_paragraph_agent),
        use_evidence_paragraph_graph=string_to_bool(args.use_evidence_paragraph_graph),
        evidence_paragraph_max_candidates=args.evidence_paragraph_max_candidates,
        evidence_paragraph_max_entity_support=args.evidence_paragraph_max_entity_support,
        paragraph_train_seed_top_k=args.paragraph_train_seed_top_k,
        paragraph_train_dpr_top_k=args.paragraph_train_dpr_top_k,
        paragraph_train_fact_ratio=args.paragraph_train_fact_ratio,
        paragraph_train_fact_seed_source=args.paragraph_train_fact_seed_source,
        paragraph_train_fact_search_seed_top_k=args.paragraph_train_fact_search_seed_top_k,
        paragraph_train_fact_search_steps=args.paragraph_train_fact_search_steps,
        paragraph_seed_mode=args.paragraph_seed_mode,
        paragraph_fact_seed_score_mode=args.paragraph_fact_seed_score_mode,
        repair_enabled=string_to_bool(args.repair_enabled),
        repair_apply_results=string_to_bool(args.repair_apply_results),
        fact_edge_dropout_rate=args.fact_edge_dropout_rate,
        fact_edge_dropout_seed=args.fact_edge_dropout_seed,
        fact_node_dropout_rate=args.fact_node_dropout_rate,
        fact_node_dropout_seed=args.fact_node_dropout_seed,
        targeted_bridge_fact_dropout=string_to_bool(args.targeted_bridge_fact_dropout),
        targeted_bridge_passage_mask=string_to_bool(args.targeted_bridge_passage_mask),
        targeted_bridge_facts_per_query=args.targeted_bridge_facts_per_query,
        query_relation_fact_dropout=string_to_bool(args.query_relation_fact_dropout),
        query_relation_passage_mask=string_to_bool(args.query_relation_passage_mask),
        query_relation_facts_per_query=args.query_relation_facts_per_query,
    )
    
    # Updated Class usage: CoEviRAG
    coevirag = CoEviRAG(global_config=config)
    coevirag.index(docs)
    if string_to_bool(args.openie_only):
        logger.info("OpenIE-only run completed; skipping graph retrieval setup and training.")
        return
    coevirag.prepare_retrieval_objects()
    if not args.policy_run_dir:
        args.policy_run_dir = os.path.join(args.save_dir, dataset_name, "models", "agents")
    if args.policy_run_dir:
        policy_run_dir = os.path.abspath(args.policy_run_dir)
        os.makedirs(policy_run_dir, exist_ok=True)
        coevirag.policy_network_path = os.path.join(policy_run_dir, "best_policy.pth")
        coevirag.paragraph_policy_network_path = os.path.join(
            policy_run_dir,
            "best_paragraph_policy.pth",
        )
        logger.info("Using isolated policy directory: %s", policy_run_dir)
    coevirag.apply_fact_node_dropout(
        rate=args.fact_node_dropout_rate,
        seed=args.fact_node_dropout_seed,
    )
    coevirag.apply_fact_edge_dropout(
        rate=args.fact_edge_dropout_rate,
        seed=args.fact_edge_dropout_seed,
    )
    if string_to_bool(args.targeted_bridge_fact_dropout):
        coevirag.apply_targeted_bridge_fact_dropout(
            queries=val_queries,
            gold_docs=val_gold_docs,
            facts_per_query=args.targeted_bridge_facts_per_query,
            seed=args.fact_edge_dropout_seed,
            mask_supporting_passages=string_to_bool(args.targeted_bridge_passage_mask),
        )
    if string_to_bool(args.query_relation_fact_dropout):
        coevirag.apply_query_relation_fact_dropout(
            queries=val_queries,
            facts_per_query=args.query_relation_facts_per_query,
            seed=args.fact_edge_dropout_seed,
            mask_source_passages=string_to_bool(args.query_relation_passage_mask),
        )
    coevirag.load_rl_agent(training=True)

    all_queries_to_embed = list(set(train_queries + val_queries))
    logger.info(f"Precomputing embeddings for {len(all_queries_to_embed)} unique queries and triplets...")
    coevirag.precompute_embeddings(all_queries_to_embed)

    # Gold embeddings are only consumed by Fact/Paragraph policy training.
    needs_gold_embeddings = (
        string_to_bool(args.train_fact_agent)
        or string_to_bool(args.train_paragraph_agent)
    )
    if needs_gold_embeddings:
        all_data_for_precompute = training_data + [
            {'query': q, 'gold_docs': gd} for q, gd in zip(val_queries, val_gold_docs)
        ]
        gold_emb_cache = precompute_gold_embeddings(coevirag, all_data_for_precompute)
    else:
        gold_emb_cache = {}
        logger.info("Skipping unused gold document embeddings for frozen retrieval evaluation.")

    # --- 3. Training and Validation Loop ---
    best_checkpoint_score = -float("inf")
    best_val_metrics = {}
    best_epoch = -1
    best_model_path = coevirag.policy_network_path
    save_policy = string_to_bool(args.save_policy)
    save_full_state = string_to_bool(args.save_training_state)
    snapshot_all_epochs = args.policy_snapshot_epochs.strip().lower() == 'all'
    snapshot_epochs = set() if snapshot_all_epochs else {int(value) for value in args.policy_snapshot_epochs.split(',') if value.strip()}
    snapshot_dir = os.path.join(os.path.dirname(best_model_path), 'epoch_snapshots')

    def save_policy_snapshot(epoch):
        if not snapshot_all_epochs and epoch not in snapshot_epochs:
            return
        os.makedirs(snapshot_dir, exist_ok=True)
        torch.save(coevirag.rl_policy_network.state_dict(),
                   os.path.join(snapshot_dir, f'epoch{epoch}_fact.pth'))
        torch.save(coevirag.paragraph_policy_network.state_dict(),
                   os.path.join(snapshot_dir, f'epoch{epoch}_paragraph.pth'))
        logger.info('Saved Fact/Paragraph policy snapshot at epoch %d', epoch)

    training_state_path = args.training_state_path or os.path.join(
        os.path.dirname(best_model_path),
        "latest_training_state.pth",
    )
    start_epoch = 1
    validation_checks_without_improvement = 0
    early_stopping_best_score = -float("inf")
    early_stopping_patience = max(0, int(args.early_stopping_patience))
    early_stopping_min_delta = max(0.0, float(args.early_stopping_min_delta))

    logger.info("--- Starting Training and Validation Loop ---")
    if args.resume_training_state:
        resumed_state = load_training_state(
            args.resume_training_state,
            coevirag,
            args.checkpoint_metric,
        )
        start_epoch = int(resumed_state["epoch"]) + 1
        best_checkpoint_score = float(
            resumed_state.get("best_checkpoint_score", -float("inf"))
        )
        best_val_metrics = dict(resumed_state.get("best_val_metrics", {}))
        best_epoch = int(resumed_state.get("best_epoch", -1))
        early_stopping_best_score = float(resumed_state.get(
            "early_stopping_best_score", best_checkpoint_score,
        ))
        validation_checks_without_improvement = int(resumed_state.get(
            "validation_checks_without_improvement", 0,
        ))
        logger.info(
            "Resumed complete training state from %s at epoch %d; "
            "continuing through epoch %d.",
            args.resume_training_state,
            start_epoch - 1,
            args.epochs,
        )
    elif string_to_bool(args.validate_before_training):
        logger.info("--- Epoch 0: Evaluating Initial Policies on Validation Set ---")
        coevirag.rl_policy_network.eval()
        coevirag.paragraph_policy_network.eval()
        with torch.no_grad():
            initial_retrieval_results, best_val_metrics = coevirag.retrieve(
                queries=val_queries,
                gold_docs=val_gold_docs,
            )
        if args.retrieval_output_json:
            save_retrieval_rankings(
                output_path=args.retrieval_output_json,
                retrieval_results=initial_retrieval_results,
                retrieval_metrics=best_val_metrics,
                metadata={
                    "dataset": args.dataset,
                    "split": args.eval_split,
                    "offset": args.val_offset,
                    "limit": len(val_queries),
                    "epoch": 0,
                    "repair_enabled": string_to_bool(args.repair_enabled),
                    "use_paragraph_agent": string_to_bool(args.use_paragraph_agent),
                    "repair_triggered_query_indices": list(
                        getattr(coevirag, "last_repair_triggered_query_indices", [])
                    ),
                },
            )
        best_checkpoint_score = get_checkpoint_score(
            best_val_metrics,
            args.checkpoint_metric,
        )
        best_epoch = 0
        early_stopping_best_score = best_checkpoint_score
        logger.info(
            "Epoch 0 checkpoint score (%s): %.6f [R@2=%.4f, R@5=%.4f, R@10=%.4f]",
            args.checkpoint_metric,
            best_checkpoint_score,
            best_val_metrics.get("Recall@2", 0.0),
            best_val_metrics.get("Recall@5", 0.0),
            best_val_metrics.get("Recall@10", 0.0),
        )
        if save_policy:
            torch.save(
                coevirag.rl_policy_network.state_dict(),
                best_model_path,
            )
            torch.save(
                coevirag.paragraph_policy_network.state_dict(),
                coevirag.paragraph_policy_network_path,
            )
            logger.info(
                "Saved epoch-0 Fact and Paragraph checkpoints before training."
            )
        save_policy_snapshot(0)
        coevirag.rl_policy_network.train()
        coevirag.paragraph_policy_network.train()

    for epoch in range(start_epoch, args.epochs + 1):
        
        if string_to_bool(args.train_fact_agent):
            train_rl_agent_epoch(coevirag, training_data, epoch, gold_emb_cache)
        else:
            logger.info(f"Epoch {epoch} [Training] Skipped; using frozen fact policy.")
        train_paragraph_agent_epoch(coevirag, training_data, epoch, gold_emb_cache)
        if getattr(coevirag, "cache_static_retrieval", False):
            logger.info(
                "Epoch %d [Static Retrieval Cache] entries=fact_scores:%d,"
                "fact_ranks:%d,dense:%d stats=%s",
                epoch,
                len(coevirag.fact_score_cache),
                len(coevirag.fact_rank_cache),
                len(coevirag.dense_retrieval_cache),
                coevirag.static_retrieval_cache_stats,
            )
        if string_to_bool(args.use_learned_coordinator):
            train_learned_coordinator(
                coevirag=coevirag,
                train_queries=train_queries,
                train_gold_docs=train_gold_docs,
                max_queries=args.coordinator_train_limit,
                max_negatives_per_query=args.coordinator_max_negatives,
                epochs=args.coordinator_epochs,
                lr=args.coordinator_lr,
                loss_type=args.coordinator_loss,
                model_type=args.coordinator_model_type,
                hidden_dim=args.coordinator_hidden_dim,
                feature_cache_path=args.coordinator_feature_cache,
            )
        
        if epoch % args.validate_every_n_epochs == 0:
            logger.info(f"--- Epoch {epoch}: Evaluating on Validation Set ---")
            
            coevirag.rl_policy_network.eval()
            coevirag.paragraph_policy_network.eval()
            
            val_metrics = {}
            with torch.no_grad():
                # Removed 'retrieval_mode' argument as per updated CoEviRAG.py
                retrieval_results, retrieval_metrics = coevirag.retrieve(
                    queries=val_queries,
                    gold_docs=val_gold_docs
                )
                val_metrics = retrieval_metrics
                if args.retrieval_output_json:
                    save_retrieval_rankings(
                        output_path=args.retrieval_output_json,
                        retrieval_results=retrieval_results,
                        retrieval_metrics=retrieval_metrics,
                        metadata={
                            "dataset": args.dataset,
                            "split": args.eval_split,
                            "offset": args.val_offset,
                            "limit": len(val_queries),
                            "epoch": epoch,
                            "repair_enabled": string_to_bool(args.repair_enabled),
                            "use_paragraph_agent": string_to_bool(args.use_paragraph_agent),
                            "repair_triggered_query_indices": list(
                                getattr(coevirag, "last_repair_triggered_query_indices", [])
                            ),
                        },
                    )
            
            current_checkpoint_score = get_checkpoint_score(
                val_metrics,
                args.checkpoint_metric,
            )
            logger.info(
                "Validation checkpoint score (%s): %.6f [R@2=%.4f, R@5=%.4f, R@10=%.4f]",
                args.checkpoint_metric,
                current_checkpoint_score,
                val_metrics.get("Recall@2", 0.0),
                val_metrics.get("Recall@5", 0.0),
                val_metrics.get("Recall@10", 0.0),
            )

            replace_checkpoint, checkpoint_improved = checkpoint_decision(
                current_checkpoint_score,
                best_checkpoint_score,
                early_stopping_best_score,
                early_stopping_min_delta,
            )
            if checkpoint_improved:
                early_stopping_best_score = current_checkpoint_score
                validation_checks_without_improvement = 0
            else:
                validation_checks_without_improvement += 1

            if replace_checkpoint:
                selection_reason = (
                    "higher score" if current_checkpoint_score > best_checkpoint_score
                    else "equal score, latest epoch"
                )
                best_checkpoint_score = current_checkpoint_score
                best_val_metrics = dict(val_metrics)
                best_epoch = epoch
                if save_policy:
                    torch.save(coevirag.rl_policy_network.state_dict(), best_model_path)
                    torch.save(coevirag.paragraph_policy_network.state_dict(), coevirag.paragraph_policy_network_path)
                    logger.info(
                        "*** Selected Checkpoint: %s. Saved to %s (Epoch %d, %s: %.6f) ***",
                        selection_reason,
                        best_model_path,
                        best_epoch,
                        args.checkpoint_metric,
                        best_checkpoint_score,
                    )
                else:
                    logger.info(
                        "*** Selected Checkpoint: %s. Checkpoint saving disabled. "
                        "(Epoch %d, %s: %.6f) ***",
                        selection_reason,
                        best_epoch,
                        args.checkpoint_metric,
                        best_checkpoint_score,
                    )
            if not checkpoint_improved and early_stopping_patience > 0:
                logger.info(
                    "Early-stopping progress: %d/%d validation checks without improvement.",
                    validation_checks_without_improvement,
                    early_stopping_patience,
                )
            
            coevirag.rl_policy_network.train()
            coevirag.paragraph_policy_network.train()

        save_policy_snapshot(epoch)
        if save_full_state:
            save_training_state(
                path=training_state_path,
                epoch=epoch,
                coevirag=coevirag,
                best_checkpoint_score=best_checkpoint_score,
                best_val_metrics=best_val_metrics,
                best_epoch=best_epoch,
                checkpoint_metric=args.checkpoint_metric,
                early_stopping_best_score=early_stopping_best_score,
                validation_checks_without_improvement=validation_checks_without_improvement,
            )
            logger.info("Saved resumable training state to %s", training_state_path)

        # Save "latest" model every 10 epochs
        if save_policy and epoch % 10 == 0:
             latest_model_path = os.path.join(os.path.dirname(best_model_path), "latest_policy.pth")
             torch.save(coevirag.rl_policy_network.state_dict(), latest_model_path)
             logger.info(f"Saved latest model checkpoint to {latest_model_path}")

        if (
                early_stopping_patience > 0
                and validation_checks_without_improvement >= early_stopping_patience):
            logger.info(
                "Early stopping at Epoch %d; best Epoch %d with %s: %.6f.",
                epoch,
                best_epoch,
                args.checkpoint_metric,
                best_checkpoint_score,
            )
            break


    logger.info(
        "Training Completed. Best model was at Epoch %d with %s: %.6f "
        "[R@2=%.4f, R@5=%.4f, R@10=%.4f]",
        best_epoch,
        args.checkpoint_metric,
        best_checkpoint_score,
        best_val_metrics.get("Recall@2", 0.0),
        best_val_metrics.get("Recall@5", 0.0),
        best_val_metrics.get("Recall@10", 0.0),
    )
    if save_policy:
        logger.info(f"Best model saved at: {best_model_path}")
    else:
        logger.info("Best model checkpoint saving was disabled for this run.")

if __name__ == "__main__":
    main()
