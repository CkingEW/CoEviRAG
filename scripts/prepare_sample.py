"""Prepare the bundled fictional sample for training, retrieval, and QA."""
import argparse
import json
from pathlib import Path

from prepare_data import write_json

ROOT = Path(__file__).resolve().parents[1]


def prepare_sample(output, fixture=ROOT / "examples/sample.json"):
    payload = json.loads(fixture.read_text(encoding="utf-8"))
    passages = payload["passages"]
    titles = {row["title"] for row in passages}
    if len(titles) != len(passages):
        raise ValueError("Sample passage titles must be unique")
    splits, ids, questions = {}, set(), set()
    for split in ("train", "val", "test"):
        rows = []
        for sample in payload["splits"][split]:
            question_key = " ".join(sample["question"].casefold().split())
            if sample["id"] in ids or question_key in questions:
                raise ValueError("Sample IDs and questions must be disjoint")
            ids.add(sample["id"])
            questions.add(question_key)
            support = set(sample["support_titles"])
            if not support or not support <= titles or not sample["reference"]:
                raise ValueError("Each sample requires valid evidence and answers")
            rows.append(dict(id=sample["id"], question=sample["question"],
                             reference=sample["reference"], contexts=[
                                 dict(**passage, is_supporting=passage["title"] in support)
                                 for passage in passages]))
        if not rows:
            raise ValueError("Sample splits must be nonempty")
        splits[split] = rows
    files = {f"{split}.json": rows for split, rows in splits.items()}
    files["sample_corpus.json"] = passages
    report = dict(dataset="sample", corpus_passages=len(passages),
                  **{split: len(rows) for split, rows in splits.items()})
    if output.exists() and any(output.iterdir()):
        try:
            matches = all((output / name).is_file() and
                          json.loads((output / name).read_text(encoding="utf-8")) == value
                          for name, value in files.items())
        except (ValueError, OSError):
            matches = False
        if matches:
            return report
        raise FileExistsError("Existing sample data differs from the bundled example; choose an empty output directory")
    for name, value in files.items():
        write_json(output / name, value)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "datasets/sample")
    args = parser.parse_args()
    print(json.dumps(prepare_sample(args.output), indent=2))


if __name__ == "__main__":
    main()
