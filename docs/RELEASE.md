# Release Scope And Limits

The source tree was read-only. `RELEASE_PROVENANCE.json` enumerates every
copied/modified source file, relative source paths and SHA-256 values, extracted
helpers, configuration evidence, and new release-owned files. No source `.git`,
outputs, weights, caches, logs, manuscript drafts, private notes, skills,
baseline programs, or abandoned-experiment runners are distributed.

Retained formal implementation: entity-fact and evidence-passage graphs; Fact
and Paragraph RL; provenance transfer; full-corpus dense safety union; fixed
RQR; DPR anchor; 53D multistate progress selector; separate learned 60D tail;
GPT-OSS-120B QA Top-5. The CLI fixes those choices. The historical label
`dynamic_setmax_recovery` controls prefix length, not RQR replanning.

Bounded release changes remove repair generation/verification/overlay methods
and prompt families, dynamic-variable-binding implementation, baseline model
factories, alternative embedding imports, vLLM OpenIE import, and fusion sweep
drivers. Feature helper functions formerly housed in an ablation script are
selected into `scripts/evidence_features.py`; no ablation driver is copied.
The unused embedding multiprocessing cache was removed because it started a
process during import; Qwen3 does not use it. Model-local lookup uses an
environment override instead of a private machine path.

Necessary shared compatibility code remains in the monolithic core:

- `_repair_normalize_entity`, `_repair_entity_aliases`,
  `_repair_looks_like_date`, and `_repair_relation_family` support adopted
  role-aware Fact traversal. Their historical names are retained.
- Repair/PPR score slots, zero counters, trace/result schemas, and empty
  repair hooks preserve shared feature/control-flow contracts. Repair hooks
  return false/empty and cannot generate, verify, or apply facts.
- Legacy fusion/coordinator/PPR branches, graph-dropout/diagnostic helpers,
  initial backend feature adapters and associated inert config fields remain
  inside shared source modules. They are not separate released methods or
  selected by `coevirag.py`. The CLI rejects arbitrary configuration keys;
  low-level Python helpers are compatibility APIs, not alternative paper
  protocols. No claim of removing every dormant line is made.

The package is `coevirag`, and the main class/module is `CoEviRAG`.
Imports, instance names, logs, CLI metadata, and generated graph-cache names
use this namespace. Network parameter names and state-dict layouts are
preserved. The original source paths in `RELEASE_PROVENANCE.json` and the
upstream attribution retain their historical names. No supplied checkpoint
or result was changed. Release validation is not an end-to-end
numerical-equivalence study. Fresh OpenIE/RQR calls and retraining can differ
from the source artifacts. The source configuration already allows validation
to retain initial policies; no claim that every selected policy improved
through RL is made.

The accompanying-code inventory claim is true for preprocessing/evaluation
utilities, configurations, ordered IDs, corpus fingerprints, and full prompt
templates/demonstrations. It does **not** imply public weights, an uploaded
repository, full raw-data byte reconstruction, or a verified supplementary
TriviaQA runtime configuration. TriviaQA is a supplementary paper dataset;
its ordered-ID adapter and fingerprints are included without rerunning it.
