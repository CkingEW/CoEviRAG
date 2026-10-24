"""Export frozen dual-agent and global dense training candidates."""
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from .features import build_candidate_features, candidate_pool


def aligned_features(components, dpr_ids, dpr_scores, content_lookup, dpr_top_k=200):
    if dpr_top_k < 1 or len(dpr_ids) != len(dpr_scores):
        raise ValueError("Invalid Dense candidate budget or unaligned scores")
    components = list(components)
    pool = candidate_pool(components)
    records = {str(row["passage"]): row for row in components}
    selected = {str(row["passage"]) for row in pool}
    for passage, score in zip(dpr_ids[:dpr_top_k], dpr_scores[:dpr_top_k]):
        passage = str(passage)
        record = records.get(passage)
        if record is None:
            record = dict(passage=passage, fact=None, paragraph=None,
                          similarity=float(score), ppr=0.0, repair=0.0,
                          agreement=0.0)
            components.append(record)
            records[passage] = record
        if passage not in selected:
            pool.append(record)
            selected.add(passage)
    base, features = build_candidate_features(components, pool)
    ids = np.asarray([str(row["passage"]) for row in pool])
    contents = [content_lookup(passage) for passage in ids]
    return base, features, ids, contents


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_training_cache(coevirag, build_components, queries, targets, args):
    root = Path(args.training_cache)
    root.mkdir(parents=True, exist_ok=True)
    identity = {
        "schema": "dual_dpr200_multiscale60_v1", "dataset": args.dataset,
        "split": args.eval_split, "seed": args.seed, "questions": queries,
        "graph_variant": args.graph_variant,
        "fact_traversal_mode": args.fact_traversal_mode,
        "fact_policy_sha256": file_digest(coevirag.policy_network_path),
        "paragraph_policy_sha256": file_digest(coevirag.paragraph_policy_network_path),
        "query_cache_sha256": file_digest(args.query_cache_override),
        "openie_sha256": file_digest(coevirag.openie_results_path)
        if hasattr(coevirag, "openie_results_path") else args.openie_results_path,
    }
    if getattr(args, "store_raw_components", False) or getattr(args, "candidate_agent_ablation", "full") != "full":
        identity.update(candidate_agent_ablation=args.candidate_agent_ablation,
                        stores_raw_components=bool(args.store_raw_components))
    manifest = root / "manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != identity:
            raise ValueError("Resume cache identity mismatch")
    else:
        manifest.write_text(json.dumps(identity, ensure_ascii=False, indent=2))
    begin = args.cache_start
    end = args.cache_stop or len(queries)
    if not 0 <= begin < end <= len(queries):
        raise ValueError("Invalid mixed cache shard interval")
    prefix_verified = False
    for query_id in range(begin, end):
        query, gold = queries[query_id], targets[query_id]
        output = root / f"{query_id:04d}.npz"
        if output.exists():
            with np.load(output) as cached:
                if str(cached["question"].item()) != query:
                    raise ValueError("Resume question mismatch")
            continue
        started = time.monotonic()
        baseline = None
        if getattr(args, "verify_fact_prefix_reuse", False) and not prefix_verified:
            coevirag.reuse_fact_path_prefix = False
            baseline = build_components(coevirag, query)
            coevirag.reuse_fact_path_prefix = True
        components = build_components(coevirag, query)
        if baseline is not None:
            def numerical_rows(rows):
                return [{key: value for key, value in row.items() if key != "info"} for row in rows]
            if numerical_rows(baseline) != numerical_rows(components):
                raise ValueError("Same-process prefix reuse changed candidates or component values")
            if [row.get("info") for row in baseline] != [row.get("info") for row in components]:
                raise ValueError("Same-process prefix reuse changed paths or evidence provenance")
            prefix_verified = True
            print(f"PREFIX_EQUIVALENCE query_id={query_id} candidates={len(components)} exact=True", flush=True)
        after_search = time.monotonic()
        dense = np.asarray(coevirag.passage_embeddings @
                           coevirag.query_to_embedding["passage"][query].T).reshape(-1)
        order = np.argsort(dense)[::-1][:600 if getattr(args, "store_raw_components", False) else 200]
        ids = [coevirag.passage_node_keys[int(index)] for index in order]
        scores = dense[order]
        base, features, passage_ids, contents = aligned_features(
            components, ids, scores,
            lambda passage: coevirag.chunk_embedding_store.get_row(passage)["content"],
        )
        if baseline is not None:
            previous = aligned_features(baseline, ids, scores,
                                        lambda passage: coevirag.chunk_embedding_store.get_row(passage)["content"])
            if not all(np.array_equal(old, new) for old, new in zip(previous[:3], (base, features, passage_ids))):
                raise ValueError("Same-process prefix reuse changed aligned features")
        after_features = time.monotonic()
        target_set = set(gold)
        with output.with_suffix(".tmp").open("wb") as file:
            extra = {}
            if getattr(args, "store_raw_components", False):
                # The numerical components retain source scores and evidence attributes; paths are not used.
                records = [{key: value for key, value in row.items() if key != "info"}
                           for row in components]
                extra["components_json"] = np.asarray(json.dumps(records, ensure_ascii=False))
                extra["dpr_ids"] = np.asarray(ids)
                extra["dpr_scores"] = np.asarray(scores, dtype=np.float32)
            np.savez_compressed(
                file, base=base, features=features, passage_ids=passage_ids,
                labels=np.asarray([content in target_set for content in contents]),
                question=np.asarray(query), query_id=np.asarray(query_id),
                **extra,
            )
        os.replace(output.with_suffix(".tmp"), output)
        if (query_id + 1) % 25 == 0 or query_id + 1 == end:
            print(f"CACHE {args.dataset} {query_id + 1}/{len(queries)}", flush=True)
            if getattr(args, "store_raw_components", False):
                timings = dict(getattr(coevirag, "last_candidate_stage_seconds", {}),
                               features=after_features-after_search,
                               save=time.monotonic()-after_features,
                               total=time.monotonic()-started)
                print("CACHE_TIMING " + json.dumps({key: round(value, 3) for key, value in timings.items()}), flush=True)
    print(f"CACHE_COMPLETE {args.dataset} range={begin}:{end}", flush=True)
