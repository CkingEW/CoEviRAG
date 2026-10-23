# Reproducibility Guide

## Configuration And Stages

`configs/final.json` selects the dataset, device, API endpoint, and training
budgets. Commands in the README use this configuration; `--dataset DATASET`
overrides the dataset for one invocation. Keep it consistent across stages.
After installation, `coevirag` and `python -m coevirag` expose the same
command interface as `python coevirag.py`.

## One-Command Workflows

```bash
python coevirag.py train --dataset hotpotqa
python coevirag.py retrieve --dataset hotpotqa
python coevirag.py qa --dataset hotpotqa
```

`train` executes the ten training stages below in order, using train and val
only. `retrieve` executes query preparation, candidate export, global dense
retrieval, safety-cache construction, and final ranking for test by default
(`--split val` selects validation). Add `--with-qa` to `retrieve` to generate
answers after retrieval. Each workflow stops immediately if a stage fails.
Use `--dry-run` to inspect the complete sequence without executing it.

Retrieval checks for both policies and both final ranking checkpoints before
preparing candidates. Training checks existing ranking histories before running
any stage; it does not silently replace an existing trained backend. Indexing
retains its existing cache behavior, but workflow stages are not automatically
skipped based on file existence. In particular, a regular `retrieve` regenerates
candidate exports and dense rankings. Use `retrieve --stage-only` to rerank
compatible existing caches, and use the individual stages to continue an
interrupted training run.

## Individual Stages

1. `index` builds the entity-fact and evidence-aware passage graphs using
   two-stage OpenIE and Qwen3 embeddings.
2. `train-policies` updates the Fact policy and then the Paragraph policy
   within each epoch. Validation selects both checkpoints, including
   initialization when it performs best.
3. `queries` prepares fixed RQRs. `export` freezes the selected policies and
   writes candidate features. Training exports include global dense Top-200;
   validation and test use the separate `dense` and `safety` stages.
4. `train-backend` trains the independent 60D unary tail and initial 51D
   selector. `train-selector` trains the final 53D multistate selector,
   including selection-progress features. Models are fitted per dataset.
5. `retrieve --stage-only` combines Fact, Passage, and global dense candidates, keeps one
   dense anchor, selects an evidence prefix, and appends the unary-ranked tail.
6. `qa` performs answer generation separately from retrieval.

The full training sequence, shown explicitly for partial reruns, is:

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

These commands use the dataset in `configs/final.json`; pass `--dataset` to
each command when overriding it. Backend scripts refuse existing training
histories rather than automatically resuming optimization. Preserve successful
stage outputs and rerun only unfinished stages; a failed ranking-training stage
may require moving its partial output directory before rerunning it. Do not mix
old candidates with newly trained policies.

The internal mode `dynamic_setmax_recovery` uses an RQR-count-derived prefix
budget, not a changing RQR. Including the anchor, its budget is
`min(4, min(4, max(2, number_of_RQR_triples)) + 1)`.

## Checkpoints And Numerical Reproduction

`configs/artifact_manifest.json` records the source checkpoints, input
fingerprints, and selected epochs. Weights, graphs, and generated caches
are not included; obtain them separately or regenerate them.

The adopted multi-hop 53D checkpoints reused earlier training-only progress
normalization statistics. To reproduce those artifacts, obtain the earlier
checkpoints and set `progress_normalization_root` to their directory. With
the default `null`, training recomputes these statistics. This provides a
retraining recipe, not identical checkpoint weights or guaranteed paper
scores. NQ used freshly computed training-only statistics.

The earlier `train_paragraph_evidence_ranker.py` and
`train_set_aware_evidence_selector.py` interfaces support existing
46D-export/initial-selector artifacts. The main pipeline uses
`train_mixed_evidence_backend.py` for independent per-dataset training.

Network parameter names and state-dict layouts are unchanged by the
CoEviRAG namespace migration. Existing policy and ranking state dictionaries
remain compatible; new graph caches use the `coevirag_fact_graph` prefix.
Remote model revisions, newly generated OpenIE/RQR outputs, and numerical
library differences can affect results. See the
[data guide](DATA.md) for ordered-ID recovery and byte-fingerprint checks.

## Generated Files

All generated artifacts belong under ignored `outputs/` directories.
`D` denotes the configured dataset and `SPLIT` the question split.

| Path | Contents |
| --- | --- |
| `outputs/D/` | OpenIE, RQRs, graph, and embedding stores |
| `outputs/policies/D/` | Selected Fact/Paragraph policies and training state |
| `outputs/backend_cache/D/train/` | Per-question training features |
| `outputs/unified_v1/D/` | Validation/test candidate components |
| `outputs/safety/SPLIT/D/` | Full-corpus dense rankings and safety cache |
| `outputs/backend/D/` | Unary tail and initial selector |
| `outputs/selector/D/` | Final multistate selector |
| `outputs/retrieval/D/` | Passage rankings and retrieval metrics |
| `outputs/qa/D/` | Answers and QA metrics |

## Offline Checks

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 MPLCONFIGDIR=/tmp/coevirag-mpl python tests/test_offline.py
python tests/test_cli.py
python coevirag.py retrieve --dry-run
```

Tests use synthetic data, CPU tensors, and mocked constructors; they do not
load pretrained models or call APIs. `tests/validate_release.py` additionally
checks a clean exported tree without Git metadata or generated files.
These checks verify software interfaces and synthetic examples, rather than
a complete end-to-end reproduction of the benchmark experiments.

## Scope And Compatibility

The main pipeline uses fixed RQRs and does not enable KG repair or dynamic
query binding. Shared feature records retain inactive compatibility fields
such as `repair` and `ppr`; some shared helper names also reflect earlier
interfaces. These fields do not add a repair stage to CoEviRAG.
