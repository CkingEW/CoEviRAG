# Reproducibility Guide

## Configuration And Stages

`configs/final.json` selects the dataset, device, API endpoint, and training
budgets. Commands in the README use this configuration; `--dataset DATASET`
overrides the dataset for one invocation. Keep it consistent across stages.
After installation, `coevirag` and `python -m coevirag` expose the same
command interface as `python coevirag.py`.

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
5. `retrieve` combines Fact, Passage, and global dense candidates, keeps one
   dense anchor, selects an evidence prefix, and appends the unary-ranked tail.
6. `qa` performs answer generation separately from retrieval.

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
python coevirag.py retrieve --dry-run
```

Tests use synthetic data, CPU tensors, and mocked constructors; they do not
load pretrained models or call APIs. `tests/validate_release.py` additionally
checks a clean exported tree without Git metadata or generated files.
The recorded checks are in `RELEASE_VALIDATION.json`; source-file provenance
is recorded in `RELEASE_PROVENANCE.json`. These checks do not establish a
complete end-to-end reproduction of the benchmark experiments.
