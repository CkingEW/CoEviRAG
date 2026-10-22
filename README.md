# CoEviRAG

Code for **CoEviRAG: Coordinated Dual-Agent Graph Navigation and Two-Stage
Evidence Ranking for Retrieval-Augmented Generation**.

CoEviRAG combines fact-graph and passage-graph navigation with an independent
dense retrieval channel. A set-conditioned selector builds complementary
evidence, and a learned ranker orders the remaining candidates.

## Installation

Use Python 3.10+ and a PyTorch build compatible with your hardware.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

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

## Data Preparation

The main pipeline supports 2WikiMultihopQA, HotpotQA, MuSiQue, and NQ.
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
See [data preparation details](docs/DATA.md) for dataset identifiers, formats,
fingerprint checks, and the supplementary TriviaQA adapter.

## Training

Run from the repository root with the same dataset configuration throughout.
These commands build the graphs, train the two agents, and train the ranking
modules on frozen agent candidates.

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

Weights and generated caches are not bundled. For checkpoint dependencies
and exact-artifact reproduction settings, see the
[reproducibility guide](docs/REPRODUCIBILITY.md).

## Retrieval

After training and validation-based model selection, run:

```bash
python coevirag.py queries --split test
python coevirag.py export --split test
python coevirag.py dense --split test
python coevirag.py safety --split test
python coevirag.py retrieve --split test
```

The `dense` and `safety` stages supply independent full-corpus candidates
alongside the two agents. Rankings and Recall@K results go to
`outputs/retrieval/`.

## QA

```bash
python coevirag.py qa --split test
```

The LLM generator answers from the first five ranked passages. Answers and
EM/F1 results go to `outputs/qa/`. See the [prompt guide](docs/PROMPTS.md)
for full templates and generation settings.

## Documentation

- [Data preparation and formats](docs/DATA.md)
- [Reproducibility, checkpoints, and offline tests](docs/REPRODUCIBILITY.md)
- [Prompt templates](docs/PROMPTS.md)

Use `python coevirag.py STAGE --dry-run` to preview a command without
executing it. The Python API is available as:

```python
from coevirag import CoEviRAG
```

## License

See [LICENSE.txt](LICENSE.txt) and [NOTICE](NOTICE) for the Apache-2.0 license
and upstream attribution, including
[HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG).
Datasets, third-party models, and hosted services retain their own terms.
