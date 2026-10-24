# Reproducibility Guide

## Configuration And Stages

`configs/final.json` selects the dataset, device, API endpoint, and training
budgets. `--dataset sample` automatically selects `configs/sample.json`;
other dataset selections use `configs/final.json`. An explicit `--config PATH`
takes precedence, and `--dataset DATASET` overrides its dataset name for one
invocation. Keep the dataset consistent across stages.
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

## Checkpoints And Normalization

`configs/artifact_manifest.json` records the paper checkpoints, input
fingerprints, and selected epochs. Pretrained checkpoints are planned for
release after paper acceptance. The current training workflow generates
weights, graphs, and caches locally.

The multi-hop 53D selectors used reference training-only progress normalization
statistics. Set `progress_normalization_root` to the directory containing the
reference selector checkpoints to reuse these statistics. For fresh training,
the default `null` computes them from the current training data. NQ used
freshly computed training-only statistics.

The main pipeline uses `scripts/train_rankers.py` for the unary tail and
initial selector, and `scripts/train_selector.py` for the final multistate
selector. Shared features, cache readers, checkpoint loaders, and selection
functions are implemented in `coevirag/ranking/`.

Network parameter names and state-dict layouts are unchanged by the
CoEviRAG namespace migration. Existing policy and ranking state dictionaries
remain compatible; new graph caches use the `coevirag_fact_graph` prefix.
See the [data guide](DATA.md) for verified dataset reconstruction,
ordered-ID recovery, and SHA-256 checks.

## Generated Files

All generated artifacts belong under `outputs/D/`, where `D` is the configured
dataset. This directory is ignored by Git. All intermediate caches are grouped
under `cache/`; `SPLIT` denotes the question split.

| Path | Contents |
| --- | --- |
| `outputs/D/cache/index/` | OpenIE, RQRs, graph, embeddings, and LLM cache |
| `outputs/D/models/agents/` | Selected Fact/Passage policies and training state |
| `outputs/D/cache/candidates/train/` | Per-question training features |
| `outputs/D/cache/candidates/SPLIT.json.gz` | Validation/test candidate components |
| `outputs/D/cache/dense/SPLIT/` | Full-corpus dense rankings and safety cache |
| `outputs/D/models/rankers/` | Unary tail and initial selector |
| `outputs/D/models/selector/` | Final multistate selector |
| `outputs/D/results/retrieval/` | Passage rankings and retrieval metrics |
| `outputs/D/results/qa/` | Answers and QA metrics |
| `outputs/D/logs/` | Saved run logs |

## Sample Test

The prepared sample is included in `datasets/sample/`, with its original
fictional source in `examples/sample.json`. Verify the sample files, train
with the one-epoch configuration, and run retrieval and QA:

```bash
python scripts/prepare_sample.py
python coevirag.py train --dataset sample
python coevirag.py retrieve --dataset sample --with-qa
```

The sample has eight passages and separate two-question train, validation,
and test splits. Its configuration trains both navigation policies, the
60D unary tail, the initial 51D selector, and the final 53D selector for one
epoch each. Retrieval uses the same fixed-RQR evidence-selection workflow,
followed by GPT-OSS-120B Top-5 answer generation. The sample-specific
`--allow_small_corpus` option uses all available passages for dense retrieval;
the four benchmark configurations retain their Top-200 requirement.

This workflow was tested successfully on the bundled sample. Its outputs are:

| Path | Contents |
| --- | --- |
| `outputs/sample/models/agents/` | Fact and Passage policy checkpoints |
| `outputs/sample/models/rankers/` | Unary tail and initial selector |
| `outputs/sample/models/selector/` | Final 53D selector |
| `outputs/sample/results/retrieval/test_metrics.json` | Recall@K metrics |
| `outputs/sample/results/retrieval/test.json` | Final ranked passages |
| `outputs/sample/results/qa/test_top5.json` | Answers and EM/F1 metrics |

Use the same model and API setup as the main workflow. The data-preparation
command verifies existing sample files or generates them in an empty directory;
it preserves locally modified data and reports a mismatch. For partial reruns,
continue with the individual stages described above.

## Scope And Compatibility

The main pipeline uses fixed RQRs and does not enable KG repair or dynamic
query binding. Shared feature records retain inactive compatibility fields
such as `repair` and `ppr`; some shared helper names also reflect earlier
interfaces. These fields do not add a repair stage to CoEviRAG.
