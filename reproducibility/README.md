# Reproducibility Materials

This directory contains the question splits, prompt snapshots, and checksums
used to document the paper experiments. It does not contain benchmark text,
model weights, or generated retrieval caches.

## Question Splits

`split_membership.json` records ordered question IDs and processed-file
fingerprints for 2WikiMultihopQA, HotpotQA, MuSiQue, NQ, and TriviaQA.
Each collection has 1,000/100/1,000 training/validation/test questions.
Training and validation use the released training partition; testing uses
the development partition.

[`prepare_data.py`](../scripts/prepare_data.py) uses these IDs to recover
the selected questions and construct a fixed corpus per collection.
See the [data guide](../docs/DATA.md) for source releases, preprocessing,
and byte-fingerprint verification.

## Prompt Snapshots

`prompts/` preserves the complete message text and demonstrations:

- `ner.py`: entity extraction from passages or questions.
- `triple_extraction.py`: entity-conditioned passage triples.
- `query_triple_extraction.py`: fixed relational query patterns.
- `answer_generation.py`: the GPT-OSS answer-generation prompt.

These files are reference snapshots. Runtime templates are in
[`coevirag/prompts/templates/`](../coevirag/prompts/templates/);
the [prompt guide](../docs/PROMPTS.md) explains their inputs and outputs.

## Checksums

`sha256.json` verifies the split manifest, this README, and the four prompt
snapshots. The offline tests check these fingerprints. Processed dataset
fingerprints are stored separately inside `split_membership.json`.
