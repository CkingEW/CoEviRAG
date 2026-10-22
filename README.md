# CoEviRAG

Code release folder for **CoEviRAG: Coordinated Dual-Agent Graph Navigation and
Two-Stage Evidence Ranking for Retrieval-Augmented Generation**.

The Python package is `coevirag`, and its main class is `CoEviRAG`:

```python
from coevirag import CoEviRAG
```

Network parameter names and state-dict layouts are unchanged. Existing policy
and ranking state dictionaries remain compatible; generated graph caches now
use the `coevirag_fact_graph` prefix.
This folder has not been uploaded or published. It includes source, offline data
adapters, configurations, ordered split IDs, corpus fingerprints, and full prompt
templates with demonstrations. It does **not** include benchmark text, pretrained
models, trained policies/rankers, extracted graphs, caches, or paper results.

## Installation

Use Python 3.10+ in an existing environment with the dependencies in
`requirements.txt`, or install in a new virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Choose a PyTorch build appropriate for your machine. `requirements-tested.txt`
records the local validation environment, not a cross-platform lock file.
The only embedding implementation is Qwen3-Embedding-4B. Obtain it separately
under its model license. Set `COEVIRAG_EMBEDDING_PATH` to a local model directory,
or let Transformers resolve `Qwen/Qwen3-Embedding-4B` when running later.

OpenIE, RQR preprocessing, and QA use a remote OpenAI-compatible GPT-OSS-120B
service. Set `OPENAI_API_KEY` in your process environment and set the endpoint
in `configs/final.json`. `.env.example` contains placeholders only; the program
does not automatically load dotenv files. Never commit credentials. No services
were called while preparing or validating this release.

## Data Preparation

Obtain the raw releases independently from the source pointers and licensing
instructions in `configs/data_sources.json`. Nothing downloads automatically.
For example, from this folder:

```bash
python scripts/prepare_data.py --dataset hotpotqa \
  --train_raw raw_datasets/hotpot_train_v1.1.json \
  --dev_raw raw_datasets/hotpot_dev_distractor_v1.json \
  --output datasets/hotpotqa
```

The same adapter supports `2wikimultihopqa`, `musique`, `mrqa_nq_fixed_1k`, and
`mrqa_triviaqa_1k`, using the raw filenames in the registry. TriviaQA is the
supplementary paper dataset in Appendix B.4; its data adapter is included, but
its supplementary runtime/checkpoint configuration was not audited in this
release task. The locked runtime CLI covers the four primary datasets.

`reproducibility/split_membership.json` is the unchanged paper snapshot. The
adapter recovers its exact **ordered identifiers**, rather than rerunning
historical reservoir sampling or requiring private historical exclusions.
Train/validation come from released training data and test from development
data: 1,000/100/1,000 questions per collection. Available multi-hop distractors
are retained. Corpus order is first occurrence of each `(title, text)` in
train, validation, then test. MRQA uses the first eligible occurrence of each
selected QID, retains the full supplied context, and preserves the source
cleaning/title/span rules. It is a fixed supplied-context adaptation, not
full-Wikipedia NQ or official full-corpus scores.

**Semantic recovery is not verified byte reproduction.** Raw releases were not
read during release validation. The adapter writes source-style UTF-8 JSON
(`ensure_ascii=False`, two-space indentation, final newline), computes all
processed hashes, and writes `reconstruction_report.json` with individual
`byte_match` results. Add `--require_byte_match` to fail on any mismatch. A
mismatch is never silently accepted as exact reproduction. Ordered IDs and
normalized questions are checked for cross-split disjointness. The historical
MuSiQue training split has 1,000 IDs but 999 distinct normalized questions;
within-split multiplicity is preserved.

Schema: each split is a JSON array of objects with `id`, `question`,
`reference: [answer, ...]`, and `contexts: [{title, text, is_supporting}, ...]`.
The corpus is `datasets/D/D_corpus.json`, an array of `{title, text}`.
Passage identity uses the exact `title + "\n" + text` string. See
`examples/schema.json` and [data contracts](docs/DATA.md).

## Reproduction Stages

Run from this folder. `configs/final.json` contains portable settings; use
`--dataset D` to select a primary dataset. Commands below are recipes for later
execution, **not actions performed in this release task**. Indexing/preprocessing
can call APIs; policy/backend training and query embedding require substantial
compute. Preview any stage with `--dry-run`, which prints a JSON command without
importing the runtime.

After installation, `coevirag` or `python -m coevirag` provides the same
command interface as `python coevirag.py`.

```bash
python coevirag.py index
python coevirag.py train-policies
python coevirag.py queries --split train
python coevirag.py queries --split val
python coevirag.py export --split train
python coevirag.py export --split val
python coevirag.py dense --split val
python coevirag.py safety --split val
python coevirag.py train-backend
python coevirag.py train-selector
```

1. Index the entity-fact graph and evidence-passage graph with two-stage OpenIE
   and Qwen3 embeddings. Passage edges are semantic/shared-evidence links, not
   verified logical entailment.
2. Train the Fact and Paragraph policies with staged REINFORCE updates. The
   source epoch loop updates Fact then Paragraph; this is not joint end-to-end
   training. Paragraph starts mix provenance-transferred Fact seeds and dense
   seeds. Validation selects checkpoints, including epoch 0 when selected.
3. Freeze both selected policies before backend feature export. Training shards
   independently score **every corpus passage** for global dense Top-200.
4. Train the independent 60D unary tail, then the initial 51D residual selector
   on frozen pools, then the final 53D multistate selector. The last stage adds
   selected-count/remaining-budget features and trains on anchor, teacher,
   frozen-policy, and recovery states. Each dataset is trained independently.

The earlier `train_paragraph_evidence_ranker.py` and
`train_set_aware_evidence_selector.py` interfaces are retained for existing
46D-export/initial-selector artifacts. The final independent pipeline above
does not need legacy comparison checkpoints. `train_mixed_evidence_backend.py`
retains its filename, but the released CLI trains one dataset at a time.

The adopted multi-hop 53D checkpoints reused earlier **training-only** progress
normalization statistics. Set `progress_normalization_root` to the supplied
earlier checkpoint directory when reproducing those exact artifacts. Without
it, the recipe recomputes training-only statistics and is a retraining recipe,
not a promise of identical weights or paper scores. See
`configs/artifact_manifest.json` for the audited dependencies and selected
epochs. NQ used freshly computed training-only progress statistics.

After freezing all model choices, prepare the test split and run retrieval:

```bash
python coevirag.py queries --split test
python coevirag.py export --split test
python coevirag.py dense --split test
python coevirag.py safety --split test
python coevirag.py retrieve --split test
```

`dense` runs **full-corpus** Qwen dot-product scoring, independent of navigation;
`safety` maps that output to original passage IDs. Both are necessary. Scoring
only graph candidates is not the safety channel. Final ranking unions
Fact200, Paragraph200, and global DPR200, keeps one DPR anchor, selects a short
set-conditioned prefix, then appends the independent learned unary tail.
The source label `dynamic_setmax_recovery` means an RQR-count-derived budget,
**not a changing RQR**: budget is
`min(4, min(4, max(2, number_of_RQR_triples)) + 1)`, including the anchor.
The final CLI rejects repair, dynamic query binding, and alternative protocols.

## QA Top-5

QA is a separate, paid-service stage and never changes retrieval:

```bash
python coevirag.py qa --split test
```

The reader consumes the first five ranked passages, uses GPT-OSS-120B,
`rag_qa_gptoss`, low reasoning effort, and a 1,024-completion-token budget. It
returns concise JSON `{"answer": "short answer"}`. See [prompt tasks and
schemas](docs/PROMPTS.md), including the actual demonstrations and the
reader's instruction to use available clues/general knowledge if needed.

## Outputs And Assets

All generated artifacts belong under ignored `outputs/`:

| Path | Contents |
| --- | --- |
| `outputs/D/` | Query RQR JSON, corpus OpenIE JSON, model-specific vector stores and graph |
| `outputs/policies/D/` | `best_policy.pth`, `best_paragraph_policy.pth`, resumable state |
| `outputs/backend_cache/D/train/` | Per-question NPZ feature shards and identity manifest |
| `outputs/unified_v1/D/` | Ordered validation/test component caches |
| `outputs/safety/SPLIT/D/` | Full-corpus dense JSON and `dpr_top200.npz` |
| `outputs/backend/D/` | `independent_tail_seed42.pth`, `independent_selector_seed42.pth` |
| `outputs/selector/D/` | Final `multistate_selector_seed42.pth`, training state records |
| `outputs/retrieval/D/` | Ranked-passage JSON and Recall@K summaries |
| `outputs/qa/D/` | QA Top-5 records and EM/F1 |

No weights or caches are packaged. The artifact manifest distinguishes local
source artifacts from included files; **local availability is not a public
download**. Obtain author artifacts separately or regenerate them. Remote
service revisions, extraction outputs, and numerical/library differences can
affect reproduction. This release makes no new result or public-weight claim.

## Offline Validation

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 MPLCONFIGDIR=/tmp/coevirag-mpl python tests/test_offline.py
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 MPLCONFIGDIR=/tmp/coevirag-mpl python tests/validate_release.py
python coevirag.py retrieve --dry-run
```

Tests use synthetic data, CPU tensors, and mocked constructors. They do not
index, train, load model weights, run a retrieval/QA experiment, or contact a
network. `RELEASE_VALIDATION.json` records the release checks. The source and
scope inventory is in `RELEASE_PROVENANCE.json` and [release notes](docs/RELEASE.md).

## License And Attribution

See `LICENSE.txt` (preserved Apache-2.0) and `NOTICE`. Graph-construction methods
are credited upstream to [HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG).
Dataset text, third-party models, and hosted services have their own terms and
are not relicensed by this folder. No repository, upload, or publication was
created as part of this handoff.
