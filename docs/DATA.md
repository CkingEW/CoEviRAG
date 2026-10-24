# Data And Cache Contracts

Public raw sources and licensing instructions are in `configs/data_sources.json`.
The adapter never downloads or embeds benchmark text in this distribution.
IDs/fingerprints are the paper snapshot; synthetic examples are in
`examples/schema.json`.

## Sources And Ordered Splits

The main runtime uses `2wikimultihopqa`, `hotpotqa`, `musique`, and
`mrqa_nq_fixed_1k`. Supplementary TriviaQA data preparation is available
through the adapter's `mrqa_triviaqa_1k` option. Supply the raw releases
listed in `configs/data_sources.json` to reconstruct each collection.

`reproducibility/split_membership.json` contains the ordered identifiers for
1,000/100/1,000 train/validation/test questions per collection. Training and
validation use released training data; testing uses development data.
The adapter recovers those identifiers instead of rerunning historical
sampling or requiring private exclusion lists. Ordered IDs and normalized
questions are checked for cross-split disjointness. The MuSiQue training
split has 1,000 IDs and 999 distinct normalized questions; within-split
multiplicity is preserved.

MRQA collections retain the full supplied context and use one labeled
support per question. They are fixed-corpus adaptations rather than
full-Wikipedia NQ or official full-corpus TriviaQA evaluations.

## Formats And Reconstruction

All split arrays are ordered. `id` is a string, `question` a string,
`reference` an array of accepted answers, and `contexts` an array of
`{title: string, text: string, is_supporting: boolean}`. NQ additionally keeps
`source` and `source_context_sha256`; MRQA TriviaQA keeps source document/QA IDs.
Source tags/whitespace are normalized by the source-derived adapter.
The corpus retains the selected contexts, including distractors, and
deduplicates them by `(title, text)` while preserving full passage text.

2Wiki/Hotpot sentence lists are concatenated without an inserted separator,
then stripped, matching `build_formal_datasets.py`. MuSiQue uses the answerable
v1.0 release paragraphs. NQ removes only uppercase `<P>`/`</P>` tags and
collapses whitespace; opaque titles use the cleaned-content SHA-256 prefix.
TriviaQA removes `[DOC]`, `[TLE]`, `[PAR]`, `[SEP]`, collapses whitespace, and
uses source-name titles or raw-content-hash titles. NQ verifies detected
answer-span content after cleaning; TriviaQA uses the source in-range span
eligibility check. The source's first valid selected-QID occurrence is used.

The corpus is the ordered union across train, val, test. Processed serialization
uses UTF-8, two-space indentation, non-ASCII preservation, and a trailing newline.
Reconstruction from the original releases was verified on 2026-10-24 for all
four main datasets. Ordered question IDs were recovered, cross-split ID and
normalized-question separation checks passed, and the SHA-256 checksums of
all three split files and the corpus matched the recorded experiment files.

| Dataset | Train / Val / Test | Corpus passages | Split and corpus SHA-256 |
| --- | --- | --- | --- |
| 2WikiMultihopQA | 1,000 / 100 / 1,000 | 11,918 | All matched |
| HotpotQA | 1,000 / 100 / 1,000 | 20,269 | All matched |
| MuSiQue | 1,000 / 100 / 1,000 | 26,475 | All matched |
| NQ | 1,000 / 100 / 1,000 | 2,076 | All matched |

The generated `reconstruction_report.json` records the expected and actual
SHA-256 checksums and each file's `byte_match`. Pass `--require_byte_match`
to validate the reconstructed files against the published fingerprints.

Query RQR JSON maps the exact question string to an object with `raw_triples`
and the source bookkeeping fields (`original_query`, `extracted_triplets`,
optional fallback reason). Corpus OpenIE JSON has `docs`, each with `idx`
(the passage hash ID), `passage`, `extracted_entities`, and
`extracted_triples`. These generated caches contain benchmark text and are
excluded from the release.

Passage Parquet stores contain `hash_id`, `content`, and `embedding`. Frozen
component gzip JSON stores one candidate-record array per split question, in
the same order. Records retain source scores (`fact`, `paragraph`, `similarity`),
agreement and separated Paragraph evidence signals. Compatibility keys `repair`
and `ppr` are zero under the final protocol.

Training shards have 14D `base`, 60D `features`, boolean `labels`, string
`passage_ids`, `question`, and integer `query_id`, plus a manifest matching
question order and input hashes. The safety NPZ has question-aligned
`passage_ids[N,200]`, finite `scores[N,200]`, `questions[N]`, a deduplicated
`contents_json` map, and metadata. A safety cache is built only from
independent full-corpus scores, never from the graph-limited candidate pool.

Final retrieval JSON has `metadata` and `results`, each with `question`,
ordered `docs`, and `scores`. The docs encode the actual selected prefix/tail
order; exported scores are original dense similarities, not calibrated final
selector scores. Do not re-sort the JSON by `scores`. Retrieval utilities
report support Recall@K; QA separately reports answer EM/F1.
