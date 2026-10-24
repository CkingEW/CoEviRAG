# CoEviRAG

Code for **CoEviRAG: Coordinated Dual-Agent Graph Navigation and Two-Stage
Evidence Ranking for Retrieval-Augmented Generation**.

CoEviRAG combines fact-graph and passage-graph navigation with an independent
dense retrieval channel. A set-conditioned selector builds complementary
evidence, and a learned ranker orders the remaining candidates.

- **Fact Navigation Agent:** explores an entity-fact graph and maps discoveries
  to source passages, supplying candidates and starts for passage exploration.
- **Passage Navigation Agent:** explores a passage graph from Fact-guided and
  dense retrieval starts.
- **Global dense channel:** adds independent full-corpus candidates to the
  deduplicated agent candidate pool.
- **Two-stage evidence ranking:** uses a dense relevance anchor, a short
  set-conditioned evidence prefix, and a learned unary ranker for the tail.

## Installation

Use Python 3.10+ and a PyTorch build compatible with your hardware. Run commands
from the repository root.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

An existing compatible conda environment can be used instead of creating a venv.

The framework uses **Qwen3-Embedding-4B** for embeddings and an
OpenAI-compatible **GPT-OSS-120B** API for OpenIE, relational query
representation (RQR), and answer generation.

```bash
export OPENAI_API_KEY="YOUR_API_KEY"
# Optional: use a local Qwen3-Embedding-4B model directory.
export COEVIRAG_EMBEDDING_PATH="/path/to/Qwen3-Embedding-4B"
```

Set the API endpoint in [`configs/sample.json`](configs/sample.json) for the
quick start, or [`configs/final.json`](configs/final.json) for benchmark
experiments. The benchmark configuration defaults to HotpotQA.
`.env` files are not loaded automatically. API-based stages may incur charges.
GPU jobs default to GPU1; set `CUDA_VISIBLE_DEVICES` to choose another device.

## Quick Start

The bundled sample contains eight fictional passages and six two-hop questions,
with two distinct questions in each training, validation, and test split.
Prepared data are included in `datasets/sample/`. After installation and API
setup, run:

```bash
python scripts/prepare_sample.py
python coevirag.py train --dataset sample
python coevirag.py retrieve --dataset sample --with-qa
```

This runs graph construction, one epoch of training for both agents and each
ranking module, final retrieval, and Top-5 answer generation. The sample uses
the same workflow as the benchmarks; dense retrieval uses all eight available
passages. Selecting `--dataset sample` automatically loads `configs/sample.json`.
The run loads Qwen3-Embedding-4B and calls the configured GPT-OSS-120B API.

Retrieval metrics and answers are written to `outputs/sample/results/retrieval/`
and `outputs/sample/results/qa/test_top5.json`. Preparation verifies the bundled files when
they already exist. Use a fresh set of sample training outputs for a new run.

## Data Preparation

For benchmark experiments, prepare one of the following datasets:

| Dataset | `--dataset` identifier |
| --- | --- |
| 2WikiMultihopQA | `2wikimultihopqa` |
| HotpotQA | `hotpotqa` |
| MuSiQue | `musique` |
| Natural Questions, MRQA fixed-corpus adaptation | `mrqa_nq_fixed_1k` |

Download the raw releases listed in
[`configs/data_sources.json`](configs/data_sources.json), then prepare a dataset:

```bash
python scripts/prepare_data.py --dataset hotpotqa \
  --train_raw raw_datasets/hotpot_train_v1.1.json \
  --dev_raw raw_datasets/hotpot_dev_distractor_v1.json \
  --output datasets/hotpotqa
```

The adapter uses the provided question IDs to recover 1,000/100/1,000
training/validation/test questions and builds a fixed corpus per dataset.
Each prepared dataset contains `train.json`, `val.json`, `test.json`, and
`DATASET_corpus.json`. Raw data, processed data, and generated model artifacts
are excluded from Git, except for the bundled `datasets/sample/` example.
See [data preparation details](docs/DATA.md) for dataset identifiers, formats,
fingerprint checks, and the supplementary TriviaQA adapter.

### Custom Datasets

Your own single-hop or multi-hop QA data can use the same normalized format
as the bundled [sample](datasets/sample/). Prepare four UTF-8 JSON files:

```text
datasets/my_dataset/
|-- train.json
|-- val.json
|-- test.json
`-- my_dataset_corpus.json
```

Each split is a JSON array of question records:

```json
[
  {
    "id": "q001",
    "question": "In which country is Aster Institute located?",
    "reference": ["Norland"],
    "contexts": [
      {
        "title": "Aster Institute",
        "text": "Aster Institute is a research institute in the city of Linden.",
        "is_supporting": true
      },
      {
        "title": "Linden",
        "text": "Linden is a city in the country of Norland.",
        "is_supporting": true
      }
    ]
  }
]
```

| Field | Type | Description |
| --- | --- | --- |
| `id` | string | Unique question identifier; use disjoint IDs across splits. |
| `question` | string | The question text. |
| `reference` | array of strings | Accepted answers or answer aliases for QA evaluation. |
| `contexts` | array of objects | Supporting passages and optional distractors. |
| `contexts[].title` | string | Passage title. |
| `contexts[].text` | string | Passage text, without the title prepended. |
| `contexts[].is_supporting` | boolean | Whether the passage is labeled as supporting evidence. |

The corpus is a separate JSON array of passages available for retrieval:

```json
[
  {
    "title": "Aster Institute",
    "text": "Aster Institute is a research institute in the city of Linden."
  },
  {
    "title": "Linden",
    "text": "Linden is a city in the country of Norland."
  }
]
```

Include the context passages from all three splits and any additional
distractors in the corpus, deduplicated by `(title, text)`. Keep titles and
texts identical between the split files and corpus: the code matches evidence
using `title + "\n" + text`. Each labeled question needs at least one supporting
passage. Training uses these passage-level labels, while QA evaluation uses
the answer aliases in `reference`. Keep normalized question texts disjoint
across splits. The benchmark dense channel expects at least 200 corpus
passages; the bundled sample enables small-corpus retrieval explicitly.

To use a new name such as `my_dataset`, add it to `DATASETS` in
`coevirag/cli.py` and `BENCHMARK_DATASETS` in `coevirag/release_protocol.py`.
The training and export stages share this registry. Use the model and training settings
in `configs/final.json` and select your dataset by name:

```bash
python coevirag.py train --dataset my_dataset
python coevirag.py retrieve --dataset my_dataset --with-qa
```

The benchmark commands use up to 1,000 training and 100 validation questions.
For larger custom splits, set `--train_limit` and `--val_limit` to `0` in the
training command generated by `coevirag/cli.py` to use all questions consistently
for agent training and candidate export. Custom data already in this format
can be used directly; `scripts/prepare_data.py` is only needed to reconstruct
the supplied benchmark splits.

## Training

One command runs graph construction, agent training, candidate preparation,
and training of the unary ranker and final evidence selector:

```bash
python coevirag.py train --dataset hotpotqa
```

The workflow uses only training and validation questions. It follows the
existing stage-wise procedure: train the agents, freeze the selected policies,
then train the ranking modules on their candidates. It does not run test QA.

Pretrained checkpoints are planned for release after paper acceptance. The
current release provides the training pipeline; weights, graphs, and caches
are generated locally. A fresh `train` run checks existing ranking-training
histories before starting. For an interrupted run, continue with the relevant
individual stages in the
[reproducibility guide](docs/REPRODUCIBILITY.md); training is not automatically
skipped or resumed. Move previous outputs before starting a fresh training run.

## Retrieval

After training, one command prepares query representations, exports agent
candidates, builds independent dense candidates, and ranks the combined pool:

```bash
python coevirag.py retrieve --dataset hotpotqa
```

The default split is `test`; use `--split val` for validation. To rerank
already prepared candidates without repeating their generation:

```bash
python coevirag.py retrieve --dataset hotpotqa --stage-only
```

Rankings and Recall@K results go to `outputs/hotpotqa/results/retrieval/`.
Changing a ranking checkpoint requires reranking; changing agents or graph
construction requires regenerating their candidates.

## Answer Generation

Generate answers from the saved passage rankings:

```bash
python coevirag.py qa --dataset hotpotqa
```

Alternatively, run retrieval and QA together:

```bash
python coevirag.py retrieve --dataset hotpotqa --with-qa
```

The LLM generator answers from up to five ranked passages. Answers and
EM/F1 results go to `outputs/hotpotqa/results/qa/`. See the [prompt guide](docs/PROMPTS.md)
for full templates and generation settings.

## Configuration And Outputs

Use `--dataset NAME` to select a dataset. Commands load `configs/final.json`
by default, or `configs/sample.json` when selecting `--dataset sample`.
The optional `--config PATH` takes precedence over this automatic selection;
`--dataset` still overrides the dataset name within the chosen configuration.
Keep the same dataset throughout training, retrieval, and QA.
After editable installation, `coevirag` and `python -m coevirag` provide the
same interface as `python coevirag.py`.

Preview every command in a workflow without loading models or calling APIs:

```bash
python coevirag.py train --dataset hotpotqa --dry-run
python coevirag.py retrieve --dataset hotpotqa --with-qa --dry-run
```

Each dataset has its own directory under `outputs/`. All intermediate caches
stay inside its `cache/` directory. For a configured dataset `D`:

| Path | Contents |
| --- | --- |
| `outputs/D/cache/index/` | Extracted facts, RQRs, graphs, embeddings, and LLM cache |
| `outputs/D/cache/candidates/` | Agent candidate components and training features |
| `outputs/D/cache/dense/` | Independent dense rankings and safety caches |
| `outputs/D/models/agents/` | Selected Fact and Passage policies and training state |
| `outputs/D/models/rankers/` | Unary ranker and preliminary selector |
| `outputs/D/models/selector/` | Final set-conditioned selector |
| `outputs/D/results/retrieval/` | Ranked passages and retrieval metrics |
| `outputs/D/results/qa/` | Generated answers and QA metrics |
| `outputs/D/logs/` | Saved run logs |

The [reproducibility guide](docs/REPRODUCIBILITY.md) documents individual
stages, intermediate caches, checkpoint dependencies, and normalization
settings used in the paper experiments.

## Project Structure

```text
CoEviRAG/
|-- coevirag.py                 # Training/retrieval/QA command entry point
|-- coevirag/
|   |-- CoEviRAG.py             # Graph construction and dual-agent navigation
|   |-- cli.py                 # Workflows and individual-stage commands
|   |-- ranking/               # Features, caches, models, selection, and training
|   |-- embedding_model/       # Qwen3 embedding interface
|   |-- information_extraction/ # Two-stage OpenIE
|   |-- llm/                   # OpenAI-compatible API interface
|   |-- prompts/               # OpenIE, RQR, and QA templates
|   `-- evaluation/            # Retrieval and QA metrics
|-- scripts/                   # Data, candidate export, training, and ranking
|-- configs/                   # Method settings, data sources, artifact metadata
|-- reproducibility/           # Split IDs, prompt snapshots, and fingerprints
|-- docs/                      # Data, protocol, and prompt documentation
|-- examples/                  # Sample source and data-format example
|-- tests/                     # Interface and synthetic CPU regression tests
|-- datasets/                  # Bundled sample; benchmark data generated locally
`-- outputs/                   # Graphs, checkpoints, caches, and results (ignored)
```

## Documentation

- [Data preparation and formats](docs/DATA.md)
- [Reproducibility, checkpoints, and sample workflow](docs/REPRODUCIBILITY.md)
- [Prompt templates](docs/PROMPTS.md)

Individual-stage commands remain available for debugging and partial reruns.
Use `python coevirag.py --help` for options. The Python API is available as:

```python
from coevirag import CoEviRAG
```

## License

See [LICENSE.txt](LICENSE.txt) for the Apache-2.0 license and [NOTICE](NOTICE)
for upstream attribution.
Datasets, third-party models, and hosted services retain their own terms.
