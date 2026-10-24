# Modified for CoEviRAG: standalone full-corpus safety export; no baseline evaluation.
"""Build independent full-corpus Qwen Top-200 rankings for the safety channel."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split_data", type=Path, required=True)
    parser.add_argument("--embedding_store", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow_small_corpus", action="store_true",
                        help="Use all available passages for a sample corpus smaller than 200")
    args = parser.parse_args()
    from coevirag.embedding_model.qwen3 import QwenEmbeddingModel
    from coevirag.prompts.linking import get_query_instruction
    from coevirag.utils.config_utils import BaseConfig
    samples = json.loads(args.split_data.read_text(encoding="utf-8"))
    frame = pd.read_parquet(args.embedding_store, columns=["hash_id", "content", "embedding"])
    if frame.empty or frame["hash_id"].duplicated().any() or frame["content"].duplicated().any():
        raise ValueError("Safety retrieval requires nonempty, unique corpus passages")
    if len(frame) < 200 and not args.allow_small_corpus:
        raise ValueError("Safety retrieval requires at least 200 passages; use --allow_small_corpus for sample data")
    top_k = min(200, len(frame))
    model = QwenEmbeddingModel(global_config=BaseConfig(embedding_batch_size=8))
    vectors = model.batch_encode([r["question"] for r in samples],
                                instruction=get_query_instruction("query_to_passage")).astype(np.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    passages = torch.from_numpy(np.stack(frame["embedding"]).astype(np.float32)).to(device)
    results = []
    # Same dot product and descending argsort as the adopted dense exporter.
    for start in range(0, len(vectors), 100):
        scores = (torch.from_numpy(vectors[start:start + 100]).to(device) @ passages.T).cpu().numpy()
        for sample, score in zip(samples[start:start + 100], scores):
            order = np.argsort(score)[::-1][:top_k]
            results.append(dict(question=sample["question"], docs=frame.iloc[order]["content"].tolist(),
                                scores=[float(score[i]) for i in order]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(metadata=dict(ranking_mode="global_qwen_dense", top_k=top_k),
                                           results=results), ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
