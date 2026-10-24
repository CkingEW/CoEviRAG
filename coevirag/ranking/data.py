"""Question labels, OpenIE metadata, and candidate-cache loading."""
import gzip
import json
from pathlib import Path
import re
import numpy as np
from .features import build_candidate_features, build_features, candidate_pool
from .selection import prepare_groups, query_budgets
from ..query_plan_evidence import valid_openie_triples, query_plan_slot_scores

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "in", "is", "it", "of", "on", "or", "that", "the", "this", "to",
    "was", "were", "with",
}


def gold_docs(samples, dataset):
    output = []
    for sample in samples:
        if "supporting_facts" in sample:
            titles = {item[0] for item in sample["supporting_facts"]}
            separator = "" if dataset.startswith("hotpotqa") else " "
            docs = [
                item[0] + "\n" + separator.join(item[1])
                for item in sample["context"] if item[0] in titles
            ]
        elif "contexts" in sample:
            docs = [
                item["title"] + "\n" + item["text"]
                for item in sample["contexts"] if item["is_supporting"]
            ]
        else:
            docs = [
                item["title"] + "\n" + item.get("text", item.get("paragraph_text", ""))
                for item in sample["paragraphs"] if item.get("is_supporting", True)
            ]
        output.append(list(set(docs)))
    return output


def normalized_terms(text):
    return {token for token in TOKEN_PATTERN.findall(str(text).lower()) if token not in STOPWORDS}


def load_openie_metadata(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    documents = payload.get("docs", []) if isinstance(payload, dict) else payload
    metadata = {}
    for document in documents:
        passage = str(document.get("passage", ""))
        title = passage.split("\n", 1)[0].strip().lower()
        entities = {
            " ".join(TOKEN_PATTERN.findall(str(entity).lower()))
            for entity in document.get("extracted_entities", [])
        }
        entities.discard("")
        relations = {
            " ".join(TOKEN_PATTERN.findall(str(triple[1]).lower()))
            for triple in document.get("extracted_triples", []) if len(triple) >= 3
        }
        relations.discard("")
        metadata[str(document["idx"])] = {
            "title": title,
            "content": passage,
            "text": passage.lower(),
            "entities": entities,
            "relations": relations,
            "tokens": normalized_terms(passage),
            "triples": valid_openie_triples(document.get("extracted_triples", [])),
        }
    return metadata


def load_query_plans(split_path, query_cache_path, query_ids):
    samples = json.loads(Path(split_path).read_text(encoding="utf-8"))
    cache = json.loads(Path(query_cache_path).read_text(encoding="utf-8"))
    return {
        int(query_id): cache.get(samples[int(query_id)]["question"], {}).get(
            "raw_triples", []
        )
        for query_id in np.unique(query_ids)
    }


def load_npz_cache(path):
    cache = np.load(path)
    exported = cache["features"].astype(np.float32)
    query_ids = cache["query_ids"].astype(np.int64)
    return {
        "base": exported[:, :14],
        "features": build_features(exported, query_ids),
        "labels": cache["labels"].astype(bool),
        "query_ids": query_ids,
        "passage_ids": cache["passage_ids"].astype(str)
        if "passage_ids" in cache.files else None,
    }


def stream_component_sets(path, expected_count):
    import ijson
    count = 0
    with gzip.open(path, "rb") as file:
        for components in ijson.items(file, "item", use_float=True):
            if count >= expected_count:
                raise ValueError("Component cache has more queries than split")
            count += 1
            yield components
            if count % 100 == 0:
                print(f"CACHE_READ {count}/{expected_count}", flush=True)
    if count != expected_count:
        raise ValueError("Component cache has fewer queries than split")


def load_component_cache(
        path, split_path, dataset, dpr_safety_cache="", dpr_safety_top_k=0,
        repair_top_k=0, repair_reference_cache="", stream=False):
    if repair_top_k > 0 and not repair_reference_cache:
        raise ValueError("repair_top_k requires a paired Repair-off reference cache")
    samples = json.loads(Path(split_path).read_text(encoding="utf-8"))
    if stream:
        component_count = len(samples)
        component_sets = stream_component_sets(path, component_count)
    else:
        with gzip.open(path, "rt", encoding="utf-8") as file:
            component_sets = json.load(file)
        component_count = len(component_sets)
        samples = samples[:component_count]
    reference_sets = None
    if repair_reference_cache:
        with gzip.open(repair_reference_cache, "rt", encoding="utf-8") as file:
            reference_sets = json.load(file)
        if len(reference_sets) != component_count:
            raise ValueError("Repair reference cache must contain the same queries")
    safety = None
    if dpr_safety_cache:
        # NpzFile indexing decompresses on every access; materialize once per run.
        with np.load(dpr_safety_cache) as archive:
            safety = {key: archive[key] for key in archive.files}
    safety_contents = None
    if safety is not None:
        if safety["passage_ids"].shape != safety["scores"].shape:
            raise ValueError("DPR safety IDs and scores have incompatible shapes")
        if "contents_json" in safety:
            safety_contents = json.loads(str(safety["contents_json"].item()))
    if safety is not None and len(safety["passage_ids"]) < component_count:
        raise ValueError("DPR safety cache has fewer queries than component cache")
    if safety is not None and "questions" in safety:
        cached_questions = safety["questions"]
        if len(cached_questions) < len(samples) or any(
                " ".join(str(question).split()) != " ".join(sample["question"].split())
                for question, sample in zip(cached_questions, samples)):
            raise ValueError("DPR safety questions do not match split order")
    targets = gold_docs(samples, dataset)
    base_rows = []
    feature_rows = []
    labels = []
    query_ids = []
    passage_ids = []
    for query_id, (components, target) in enumerate(zip(component_sets, targets)):
        components = list(components)
        pool = candidate_pool(components)
        if repair_top_k > 0:
            selected = {str(item["passage"]) for item in pool}
            reference_ids = (
                {str(item["passage"]) for item in reference_sets[query_id]}
                if reference_sets is not None else None
            )
            repair_candidates = sorted(
                (item for item in components if (
                    str(item["passage"]) not in reference_ids
                    if reference_ids is not None else float(item.get("repair") or 0) > 0
                )),
                key=lambda item: (float(item["repair"]), str(item["passage"])),
                reverse=True,
            )[:repair_top_k]
            for item in repair_candidates:
                passage = str(item["passage"])
                if passage not in selected:
                    pool.append(item)
                    selected.add(passage)
        if safety is not None and dpr_safety_top_k > 0:
            by_passage = {str(item["passage"]): item for item in components}
            selected = {str(item["passage"]) for item in pool}
            width = min(dpr_safety_top_k, safety["passage_ids"].shape[1])
            for rank in range(width):
                passage = str(safety["passage_ids"][query_id, rank])
                record = by_passage.get(passage)
                if record is None:
                    record = {
                        "passage": passage,
                        "fact": None,
                        "paragraph": None,
                        "similarity": float(safety["scores"][query_id, rank]),
                        "ppr": 0.0,
                        "repair": 0.0,
                        "agreement": 0.0,
                        "content": (safety_contents[passage] if safety_contents is not None
                                    else str(safety["contents"][query_id, rank])),
                    }
                    components.append(record)
                    by_passage[passage] = record
                if passage not in selected:
                    pool.append(record)
                    selected.add(passage)
        base, features = build_candidate_features(components, pool)
        target_set = set(target)
        contents = [item.get("content") or item.get("info", {}).get("content") for item in pool]
        base_rows.append(base)
        feature_rows.append(features)
        labels.extend(content in target_set for content in contents)
        query_ids.extend([query_id] * len(pool))
        passage_ids.extend(str(item["passage"]) for item in pool)
    return {
        "base": np.concatenate(base_rows),
        "features": np.concatenate(feature_rows),
        "labels": np.asarray(labels, dtype=bool),
        "query_ids": np.asarray(query_ids, dtype=np.int64),
        "passage_ids": np.asarray(passage_ids, dtype=str),
    }


def gold_denominators(split_path, query_ids):
    samples = json.loads(Path(split_path).read_text(encoding="utf-8"))
    denominators = {}
    for query_id in np.unique(query_ids):
        sample = samples[int(query_id)]
        if "contexts" in sample:
            count = sum(bool(item.get("is_supporting")) for item in sample["contexts"])
        elif "supporting_facts" in sample:
            count = len({item[0] for item in sample["supporting_facts"]})
        else:
            count = sum(bool(item.get("is_supporting", True)) for item in sample["paragraphs"])
        denominators[int(query_id)] = max(1, count)
    return denominators


def assert_split_separation(datasets):
    train, val = set(), set()
    for dataset in datasets:
        for split, destination in (("train", train), ("val", val)):
            samples = json.loads(Path(f"datasets/{dataset}/{split}.json").read_text())
            questions = [" ".join(row["question"].lower().split()) for row in samples]
            if len(questions) != len(set(questions)):
                print(f"DUPLICATE_QUESTIONS {dataset}/{split} "
                      f"entries={len(questions)} unique={len(set(questions))}", flush=True)
            destination.update(questions)
    if train & val:
        raise ValueError("Train/validation question overlap across datasets")


def load_shards(root, split_path):
    manifest = json.loads((root / "manifest.json").read_text())
    samples = json.loads(split_path.read_text())
    if manifest["questions"] != [row["question"] for row in samples]:
        raise ValueError("Training manifest differs from split")
    rows = {key: [] for key in ("base", "features", "labels", "passage_ids", "query_ids")}
    for query_id, sample in enumerate(samples):
        with np.load(root / f"{query_id:04d}.npz") as shard:
            if str(shard["question"].item()) != sample["question"]:
                raise ValueError("Training shard question mismatch")
            for key in ("base", "features", "labels", "passage_ids"):
                rows[key].append(shard[key])
            rows["query_ids"].append(np.full(len(shard["labels"]), query_id, dtype=np.int64))
    return {key: np.concatenate(value) for key, value in rows.items()}


def load_bundle(dataset, output_root, train_limit, val_limit):
    train_path, val_path = [Path(f"datasets/{dataset}/{split}.json") for split in ("train", "val")]
    cache = Path(output_root) / dataset / "cache"
    query_path = cache / "index/query_triplets_cache.json"
    metadata = load_openie_metadata(cache / "index/openie_results_ner_openai_gpt-oss-120b.json")
    train = load_shards(cache / "candidates/train", train_path)
    val = load_component_cache(
        cache / "candidates/val.json.gz", val_path, dataset,
        dpr_safety_cache=cache / "dense/val/dpr_top200.npz",
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
