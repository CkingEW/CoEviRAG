"""Map frozen global Dense rankings to the original corpus IDs for safety retrieval."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def build_cache(retrieval_path, split_path, embedding_store, output, top_k, allow_small_corpus=False):
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    samples = json.loads(split_path.read_text())
    rankings = json.loads(retrieval_path.read_text())["results"]
    if len(samples) != len(rankings):
        raise ValueError("Dense rankings and split must contain the same number of queries")
    frame = pd.read_parquet(embedding_store, columns=["hash_id", "content"])
    if frame.empty:
        raise ValueError("The corpus must contain at least one passage")
    if allow_small_corpus:
        top_k = min(top_k, len(frame))
    by_content = dict(zip(frame["content"], frame["hash_id"]))
    if len(by_content) != len(frame) or frame["hash_id"].duplicated().any():
        raise ValueError("Corpus content and IDs must be unique")
    ids, scores, questions = [], [], []
    contents = {}
    for sample, row in zip(samples, rankings):
        if " ".join(sample["question"].split()) != " ".join(row["question"].split()):
            raise ValueError("Dense rankings do not match split question order")
        docs = row["docs"][:top_k]
        values = row["scores"][:top_k]
        if len(docs) != top_k or len(values) != top_k or len(set(docs)) != top_k:
            raise ValueError("Dense rankings must provide top_k unique scored passages")
        if not np.isfinite(values).all():
            raise ValueError("Dense scores must be finite")
        passage_ids = [str(by_content[doc]) for doc in docs]
        contents.update(zip(passage_ids, docs))
        ids.append(passage_ids)
        scores.append(values)
        questions.append(sample["question"])
    metadata = {"retrieval": str(retrieval_path), "split_data": str(split_path),
                "embedding_store": str(embedding_store), "top_k": top_k}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    # Store variable-length text once per passage, without padded 2D string arrays.
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, passage_ids=np.asarray(ids),
                            scores=np.asarray(scores, dtype=np.float32),
                            questions=np.asarray(questions),
                            contents_json=np.asarray(json.dumps(contents, ensure_ascii=False)),
                            metadata_json=np.asarray(json.dumps(metadata)))
    os.replace(temporary, output)
    return {"queries": len(samples), "top_k": top_k, "unique_passages": len(contents)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--split_data", type=Path, required=True)
    parser.add_argument("--embedding_store", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top_k", type=int, default=200)
    parser.add_argument("--allow_small_corpus", action="store_true",
                        help="Use all available passages for a sample corpus smaller than top_k")
    args = parser.parse_args()
    print(json.dumps(build_cache(args.retrieval, args.split_data, args.embedding_store,
                                args.output, args.top_k, args.allow_small_corpus)))


if __name__ == "__main__":
    main()
