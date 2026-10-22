# Modified for CoEviRAG: frozen candidate export only; no sweeps.
import argparse

import gzip

import json

import logging

import os

import random

import sys

import time

from pathlib import Path

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

import torch

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from coevirag.CoEviRAG import CoEviRAG

from coevirag.prompts.linking import get_query_instruction

from coevirag.utils.config_utils import BaseConfig

from coevirag.utils.misc_utils import min_max_normalize

logger = logging.getLogger(__name__)

K_VALUES = [1, 2, 5, 10, 20, 30, 50, 100, 150, 200]

def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

def get_gold_docs(samples: List[Any], dataset_name: str) -> List[List[str]]:
    gold_docs = []
    for sample in samples:
        if "supporting_facts" in sample:
            gold_titles = {item[0] for item in sample["supporting_facts"]}
            supporting = [item for item in sample["context"] if item[0] in gold_titles]
            separator = "" if dataset_name.startswith("hotpotqa") else " "
            docs = [item[0] + "\n" + separator.join(item[1]) for item in supporting]
        elif "contexts" in sample:
            docs = [
                item["title"] + "\n" + item["text"]
                for item in sample["contexts"]
                if item["is_supporting"]
            ]
        else:
            docs = [
                item["title"] + "\n" + item.get("text", item.get("paragraph_text"))
                for item in sample["paragraphs"]
                if item.get("is_supporting", True)
            ]
        gold_docs.append(list(set(docs)))
    return gold_docs

def build_components(
        coevirag: CoEviRAG,
        query: str) -> List[Dict]:
    started = time.monotonic()
    query_fact_scores = coevirag.get_fact_scores(query)
    top_k_fact_indices, top_k_facts = coevirag.rank_facts(query, query_fact_scores)
    after_linking = time.monotonic()
    query_emb_tensor = coevirag._get_query_embedding_tensor(query)
    dpr_ids, _ = coevirag.dense_passage_retrieval(query)

    if top_k_facts:
        fact_info, query_emb_raw = coevirag.run_rl_graph_search_rerank(
            query,
            top_k_facts,
            top_k_fact_indices=top_k_fact_indices,
            query_fact_scores=query_fact_scores,
        )
        if isinstance(query_emb_raw, np.ndarray):
            query_emb_tensor = torch.from_numpy(query_emb_raw).float().to(
                coevirag.device
            ).view(1, -1)
        else:
            query_emb_tensor = query_emb_raw.to(coevirag.device).view(1, -1)
    else:
        fact_info = {}
    after_fact = time.monotonic()

    ranked_fact_seed_items = sorted(
        fact_info.items(),
        key=lambda item: (item[1].get("score", -float("inf")), item[0]),
        reverse=True,
    )[:getattr(coevirag.global_config, "paragraph_agent_seed_top_k", 20)]
    ranked_fact_seed_items = coevirag._gate_fact_seeds_for_paragraph(
        ranked_fact_seed_items,
        query_emb_tensor,
    )
    fact_seed_scores = (
        min_max_normalize(np.asarray([
            float(info.get("score", 0.0))
            for _, info in ranked_fact_seed_items
        ]))
        if ranked_fact_seed_items else np.asarray([], dtype=np.float32)
    )
    fact_seed_passages = [
        (passage, float(score))
        for (passage, _), score in zip(ranked_fact_seed_items, fact_seed_scores)
    ]
    paragraph_info = coevirag.run_paragraph_graph_search(
        query=query, seed_passages=fact_seed_passages, query_emb=query_emb_tensor)
    after_paragraph = time.monotonic()
    repair_info = {}
    passage_info = coevirag._merge_agent_passage_info(
        fact_info,
        paragraph_info,
        repair_info,
    )
    components = coevirag._build_candidate_components(
        passage_info=passage_info,
        query_emb_tensor=query_emb_tensor,
    )
    coevirag.last_candidate_stage_seconds = {
        "linking": after_linking - started,
        "fact_search": after_fact - after_linking,
        "paragraph_search": after_paragraph - after_fact,
        "components": time.monotonic() - after_paragraph,
    }
    return components


def main():
    parser = argparse.ArgumentParser(description="Export frozen dual-agent candidates, without sweeps")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--eval_split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--policy_run_dir", required=True)
    parser.add_argument("--llm_base_url", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--candidate_cache", default="")
    parser.add_argument("--export_mixed_backend_cache", default="")
    parser.add_argument("--prepare_queries_only", action="store_true")
    args = parser.parse_args()
    from coevirag.release_protocol import DATASETS
    if args.dataset not in DATASETS:
        parser.error("Unsupported final-method dataset")
    if not args.prepare_queries_only and bool(args.candidate_cache) == bool(args.export_mixed_backend_cache):
        parser.error("Choose exactly one export format")
    set_random_seed(args.seed)
    args.graph_variant, args.fact_traversal_mode = "unified_v1", "role_aware"
    args.query_cache_override = f"outputs/{args.dataset}/query_triplets_cache.json"
    args.mixed_cache_start, args.mixed_cache_stop = 0, 0
    args.openie_results_path = ""
    data = PROJECT_ROOT / "datasets" / args.dataset
    samples = json.loads((data / f"{args.eval_split}.json").read_text(encoding="utf-8"))
    corpus = json.loads((data / f"{args.dataset}_corpus.json").read_text(encoding="utf-8"))
    config = BaseConfig(
        dataset=args.dataset, save_dir=f"outputs/{args.dataset}",
        llm_name="openai/gpt-oss-120b", llm_base_url=args.llm_base_url,
        llm_reasoning_effort="low", max_new_tokens=1024,
        embedding_model_name="Qwen/Qwen3-Embedding-4B", embedding_batch_size=8,
        graph_variant="unified_v1", execution_device=args.device,
        seed=args.seed, retrieval_top_k=200, linking_top_k=10,
        rl_seed_top_k=6, query_fact_score_mode="concat", w_query=0.0,
        cache_static_retrieval=True, batch_fact_beam_scoring=True,
        fact_traversal_mode="role_aware", rerank_fusion_mode="rank_rrf",
        rank_fusion_rrf_k=0, fact_head_protect_k=0, similarity_head_protect_k=2,
        w_policy=0.2, w_fact_policy=0.25, w_paragraph_policy=0.10, w_sim=0.60,
        w_repair_confidence=0.10, w_source_agreement=0.05,
        use_ppr_rerank=False, w_ppr=0.0, use_paragraph_agent=True,
        train_paragraph_agent=False, use_evidence_paragraph_graph=True,
        paragraph_seed_mode="hybrid", paragraph_fact_seed_score_mode="normalized",
        repair_enabled=False, repair_apply_results=False,
    )
    rag = CoEviRAG(global_config=config)
    rag.index([row["title"] + "\n" + row["text"] for row in corpus])
    rag.prepare_retrieval_objects()
    queries = [sample["question"] for sample in samples]
    # Preserve the formal exporter's first-train-query preprocessing context.
    first_train = json.loads((data / "train.json").read_text(encoding="utf-8"))[0]["question"]
    rag.precompute_embeddings(list(set([first_train] + queries)))
    if args.prepare_queries_only:
        return
    rag.policy_network_path = str(Path(args.policy_run_dir) / "best_policy.pth")
    rag.paragraph_policy_network_path = str(Path(args.policy_run_dir) / "best_paragraph_policy.pth")
    if not all(Path(p).is_file() for p in (rag.policy_network_path, rag.paragraph_policy_network_path)):
        raise FileNotFoundError("Both frozen policy checkpoints are required")
    rag.load_rl_agent(training=False)
    if args.export_mixed_backend_cache:
        from export_mixed_backend_cache import export_cache
        export_cache(rag, build_components, queries, get_gold_docs(samples, args.dataset), args)
    else:
        rows = []
        for query in tqdm(queries, desc="Exporting frozen components"):
            components = build_components(rag, query)
            for row in components:
                row["content"] = rag.chunk_embedding_store.get_row(row["passage"])["content"]
                row.pop("info", None)
            rows.append(components)
        target = Path(args.candidate_cache)
        target.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(target, "wt", encoding="utf-8") as stream:
            json.dump(rows, stream, ensure_ascii=False)


if __name__ == "__main__":
    main()
