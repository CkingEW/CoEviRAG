# Data And Cache Contracts

Public raw sources and licensing instructions are in `configs/data_sources.json`.
The adapter never downloads or embeds benchmark text in this distribution.
IDs/fingerprints are the paper snapshot; synthetic examples are in
`examples/schema.json`.

All split arrays are ordered. `id` is a string, `question` a string,
`reference` an array of accepted answers, and `contexts` an array of
`{title: string, text: string, is_supporting: boolean}`. NQ additionally keeps
`source` and `source_context_sha256`; MRQA TriviaQA keeps source document/QA IDs.
Source tags/whitespace are normalized exactly by the source-derived adapter.
The `(title, text)` corpus deduplication keeps distractors. No gold passages
are omitted or answer-centered crops introduced by the adapter.

2Wiki/Hotpot sentence lists are concatenated without an inserted separator,
then stripped, matching `build_formal_datasets.py`. MuSiQue uses the answerable
v1.0 release paragraphs. NQ removes only uppercase `<P>`/`</P>` tags and
collapses whitespace; opaque titles use the cleaned-content SHA-256 prefix.
TriviaQA removes `[DOC]`, `[TLE]`, `[PAR]`, `[SEP]`, collapses whitespace, and
uses source-name titles or raw-content-hash titles. NQ verifies detected
answer-span content after cleaning; TriviaQA uses the source in-range span
eligibility check. The source's first valid selected-QID occurrence is used.

The corpus is the ordered union across train, val, test. Processed serialization
matches the source writers' UTF-8, two-space indentation, non-ASCII preservation,
and trailing newline. Source serializer/cleaning parity is checked on synthetic
fixtures, not full benchmark data. Full processed split/corpus byte equality
has **not** been verified. The adapter reports expected versus actual SHA-256
per file and can require all byte matches. Changed upstream releases, record
occurrence differences, and transformed raw archives must not be described as
byte-identical just because IDs match.

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
