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

Set the API endpoint and dataset in [`configs/final.json`](configs/final.json).
The default dataset is HotpotQA. `.env` files are not loaded automatically.
API-based stages may incur charges.
GPU jobs default to GPU1; set `CUDA_VISIBLE_DEVICES` to choose another device.

## Data Preparation

The main pipeline supports these datasets:

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
are excluded from Git.
See [data preparation details](docs/DATA.md) for dataset identifiers, formats,
fingerprint checks, and the supplementary TriviaQA adapter.

## Training

One command runs graph construction, agent training, candidate preparation,
and training of the unary ranker and final evidence selector:

```bash
python coevirag.py train --dataset hotpotqa
```

The workflow uses only training and validation questions. It follows the
existing stage-wise procedure: train the agents, freeze the selected policies,
then train the ranking modules on their candidates. It does not run test QA.

Weights and caches are generated locally and are not bundled. A fresh `train`
run refuses existing ranking-training histories before starting. For an
interrupted run, continue with the relevant individual stages in the
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

Rankings and Recall@K results go to `outputs/retrieval/hotpotqa/`.
Changing a ranking checkpoint requires reranking; changing agents or graph
construction requires regenerating their candidates.

## QA

```bash
python coevirag.py qa --dataset hotpotqa
```

Alternatively, run retrieval and QA together:

```bash
python coevirag.py retrieve --dataset hotpotqa --with-qa
```

The LLM generator answers from up to five ranked passages. Answers and
EM/F1 results go to `outputs/qa/hotpotqa/`. See the [prompt guide](docs/PROMPTS.md)
for full templates and generation settings.

## Project Structure

```text
CoEviRAG/
|-- coevirag.py                  # Training/retrieval/QA command entry point
|-- coevirag/
|   |-- CoEviRAG.py              # Graph construction and dual-agent navigation
|   |-- cli.py                  # Workflows and individual-stage commands
|   |-- embedding_model/        # Qwen3 embedding interface
|   |-- information_extraction/ # Two-stage OpenIE
|   |-- llm/                    # OpenAI-compatible API interface
|   |-- prompts/                # OpenIE, RQR, and QA templates
|   `-- evaluation/             # Retrieval and QA metrics
|-- scripts/                    # Data, candidate export, training, and ranking
|-- configs/                    # Method settings, data sources, artifact metadata
|-- reproducibility/            # Split IDs, prompt snapshots, and fingerprints
|-- docs/                       # Data, protocol, and prompt documentation
|-- examples/                   # Synthetic data-format example
|-- tests/                      # Offline interface and synthetic CPU tests
|-- datasets/                   # Prepared data (generated locally, ignored)
`-- outputs/                    # Graphs, checkpoints, caches, and results (ignored)
```

## Configuration And Outputs

All commands accept `--config PATH`; `--dataset` overrides the dataset for one
invocation. Keep the same dataset throughout training, retrieval, and QA.
After editable installation, `coevirag` and `python -m coevirag` provide the
same interface as `python coevirag.py`.

Preview every command in a workflow without loading models or calling APIs:

```bash
python coevirag.py train --dataset hotpotqa --dry-run
python coevirag.py retrieve --dataset hotpotqa --with-qa --dry-run
```

For a configured dataset `D`:

| Path | Contents |
| --- | --- |
| `outputs/D/` | Extracted facts, RQRs, graphs, and embeddings |
| `outputs/policies/D/` | Selected Fact and Passage policies |
| `outputs/backend/D/` | Unary ranker and preliminary selector |
| `outputs/selector/D/` | Final set-conditioned selector |
| `outputs/retrieval/D/` | Ranked passages and retrieval metrics |
| `outputs/qa/D/` | Generated answers and QA metrics |

The [reproducibility guide](docs/REPRODUCIBILITY.md) documents individual
stages, intermediate caches, checkpoint dependencies, and exact-artifact
settings. Retraining and new LLM extractions can change numerical results;
the source release does not include the paper's weights or generated caches.

## Offline Checks

```bash
python tests/test_cli.py
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 MPLCONFIGDIR=/tmp/coevirag-mpl python tests/test_offline.py
```

These tests use command mocks and synthetic CPU examples, not model downloads,
API calls, or benchmark experiments.

## Documentation

- [Data preparation and formats](docs/DATA.md)
- [Reproducibility, checkpoints, and offline tests](docs/REPRODUCIBILITY.md)
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
