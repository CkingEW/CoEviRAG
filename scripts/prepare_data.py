# Modified for CoEviRAG: ordered-manifest adapter preserving source normalization.
"""Recover ordered paper splits from user-supplied public releases, offline."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = {
    "2wikimultihopqa": ("2Wiki", "multihop"),
    "hotpotqa": ("HotpotQA", "multihop"),
    "musique": ("MuSiQue", "multihop"),
    "mrqa_nq_fixed_1k": ("NQ", "nq"),
    "mrqa_triviaqa_1k": ("TriviaQA", "trivia"),
}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def records(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        if path.name.endswith((".jsonl", ".jsonl.gz")):
            for line in stream:
                if line.strip():
                    yield json.loads(line)
        else:
            yield from json.load(stream)


def normalize_multihop(dataset, row):
    # Same normalization as build_formal_datasets.py, including sentence joining.
    answers = []
    for answer in [row.get("answer"), *row.get("answer_aliases", [])]:
        if answer is not None and str(answer) not in answers:
            answers.append(str(answer))
    contexts = []
    if dataset in {"2wikimultihopqa", "hotpotqa"}:
        titles = {str(item[0]) for item in row.get("supporting_facts", [])}
        for title, sentences in row.get("context", []):
            contexts.append(dict(title=str(title), text="".join(map(str, sentences)).strip(),
                                 is_supporting=str(title) in titles))
    else:
        for item in row.get("paragraphs", []):
            contexts.append(dict(title=str(item["title"]),
                                 text=str(item.get("paragraph_text", item.get("text", ""))).strip(),
                                 is_supporting=bool(item.get("is_supporting", False))))
    if not contexts or not any(item["is_supporting"] for item in contexts):
        raise ValueError("Selected record has no supporting context")
    return dict(id=str(row.get("_id", row.get("id"))), question=str(row["question"]),
                reference=answers, contexts=contexts)


def clean_nq(raw):
    return " ".join(re.sub(r"</?P>", " ", raw).split())


def normalize_mrqa(kind, row, qa, partition):
    raw = row["context"]
    clean = clean_nq if kind == "nq" else lambda text: " ".join(
        re.sub(r"\[(?:DOC|TLE|PAR|SEP)\]", " ", text).split())
    text = clean(raw)
    answers = list(dict.fromkeys(a.strip() for a in qa["answers"] if a.strip()))
    spans = [span for item in qa.get("detected_answers", []) for span in item.get("char_spans", [])]
    valid = any(0 <= start <= end < len(raw) and (
        kind != "nq" or clean(raw[start:end + 1]).casefold() in
        {clean(a).casefold() for a in answers}) for start, end in spans)
    if not answers or not valid or (kind == "nq" and len(text) < 40):
        return None
    result = dict(id=str(qa["qid"]), question=qa["question"].strip(), reference=answers)
    if kind == "nq":
        title = "NQ passage " + hashlib.sha256(text.encode()).hexdigest()[:16]
        result.update(source="MRQA NaturalQuestionsShort " + partition,
                      source_context_sha256=hashlib.sha256(raw.encode()).hexdigest())
    else:
        source_id = str(row["id"])
        name = source_id.split("--", 1)[-1].removesuffix(".txt")
        title = (name.replace("_", " ") if "/" not in name and not name.isdigit()
                 else "TriviaQA web document " + hashlib.sha256(raw.encode()).hexdigest()[:12])
        result.update(source_document_id=source_id, source_qa_id=str(qa["id"]))
    result["contexts"] = [dict(title=title, text=text, is_supporting=True)]
    return result


def extract(path, dataset, kind, requested, partition):
    found = {}
    header_seen = False
    for row in records(path):
        if "header" in row:
            expected = "NaturalQuestionsShort" if kind == "nq" else "TriviaQA-web"
            if row["header"] != {"dataset": expected, "split": partition}:
                raise ValueError("MRQA header mismatch")
            header_seen = True
            continue
        if kind == "multihop":
            qid = str(row.get("_id", row.get("id")))
            if qid in requested and qid not in found and row.get("answerable", True):
                found[qid] = normalize_multihop(dataset, row)
        else:
            if not header_seen:
                raise ValueError("MRQA source must start with its header")
            for qa in row["qas"]:
                qid = str(qa["qid"])
                if qid in requested and qid not in found:
                    value = normalize_mrqa(kind, row, qa, partition)
                    if value is not None:
                        found[qid] = value
    missing = requested - set(found)
    if missing:
        raise ValueError(f"Source release misses {len(missing)} selected eligible IDs")
    return found


def recover(dataset, train_path, dev_path, output, manifest_path, require_hashes=False):
    label, kind = REGISTRY[dataset]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))[label]
    ids = {split: info["question_ids"] for split, info in manifest["splits"].items()}
    if any(len(order) != len(set(order)) for order in ids.values()):
        raise ValueError("Duplicate IDs in split manifest")
    if any(set(ids[a]) & set(ids[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("Split IDs overlap")
    training = extract(train_path, dataset, kind, set(ids["train"] + ids["val"]), "train")
    development = extract(dev_path, dataset, kind, set(ids["test"]), "dev")
    splits = {split: [(development if split == "test" else training)[qid] for qid in order]
              for split, order in ids.items()}
    question_keys = {split: {" ".join(row["question"].strip().split()).casefold() for row in rows}
                     for split, rows in splits.items()}
    if any(question_keys[a] & question_keys[b]
           for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("Normalized questions overlap across splits")
    corpus, seen = [], set()
    for split in ("train", "val", "test"):
        for sample in splits[split]:
            for context in sample["contexts"]:
                key = context["title"], context["text"]
                if key not in seen:
                    seen.add(key)
                    corpus.append(dict(title=key[0], text=key[1]))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose an empty output directory")
    checks = {}
    for split, samples in splits.items():
        path = output / f"{split}.json"
        write_json(path, samples)
        checks[split] = dict(actual=digest(path), expected=manifest["splits"][split]["processed_file_sha256"])
    write_json(output / f"{dataset}.json", sum(splits.values(), []))
    corpus_path = output / f"{dataset}_corpus.json"
    write_json(corpus_path, corpus)
    checks["corpus"] = dict(actual=digest(corpus_path), expected=manifest["processed_corpus_sha256"])
    for check in checks.values():
        check["byte_match"] = check["actual"] == check["expected"]
    report = dict(dataset=dataset, ordered_ids_recovered=True, corpus_count=len(corpus),
                  all_byte_fingerprints_match=all(c["byte_match"] for c in checks.values()),
                  fingerprints=checks, raw_sha256=dict(train=digest(train_path), dev=digest(dev_path)))
    write_json(output / "reconstruction_report.json", report)
    if require_hashes and not report["all_byte_fingerprints_match"]:
        raise ValueError("IDs recovered, but byte fingerprints differ; inspect reconstruction_report.json")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=REGISTRY, required=True)
    parser.add_argument("--train_raw", type=Path, required=True)
    parser.add_argument("--dev_raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "reproducibility/split_membership.json")
    parser.add_argument("--require_byte_match", action="store_true")
    args = parser.parse_args()
    print(json.dumps(recover(args.dataset, args.train_raw, args.dev_raw, args.output,
                             args.manifest, args.require_byte_match), indent=2))


if __name__ == "__main__":
    main()
