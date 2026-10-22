# Prompt Tasks And Schemas

The byte-exact paper snapshots, including demonstrations, are in
`reproducibility/prompts/`; their original SHA-256 index is retained. Runtime
templates live in `coevirag/prompts/templates/`. Message strings were not
rewritten for this release. The runtime GPT-OSS alias is resolved to the
identical adopted answer messages; no alternative model prompt is loaded.

| Task | Input | Expected JSON | Messages |
| --- | --- | --- | --- |
| Passage NER | Complete `title + newline + passage` | `{"named_entities": ["entity", ...]}` | System, one user/assistant demonstration, target user |
| Entity-conditioned triple OpenIE | Complete passage plus NER JSON | `{"triples": [["subject", "relation", "object"], ...]}` | System, one user/assistant demonstration using the NER example, target user |
| Question NER | Original question, using the same NER template | `{"named_entities": [...]}` | Same one-shot NER messages |
| Fixed RQR | Question plus its entity JSON | `{"triples": [["known entity", "relation", "?"], ...]}` | System, two user/assistant demonstrations, target user |
| QA | First five ranked passages and original question | `{"answer": "short answer"}` | System and target user, no QA demonstrations |

The NER demonstration is Radio City. The two RQR demonstrations concern
comparing directors' death dates and the director of a Tom Hanks film. Their
full text is included, not replaced by summaries or ellipses. The NER system
text says "JSON list" while its demonstration and parser use the
`named_entities` object above; this source wording is preserved verbatim.

`${passage}` is substituted with a passage or question, and
`${named_entity_json}` with a JSON object containing `named_entities`.
`${context}` and `${question}` are the reader placeholders. `?` means an
unspecified endpoint, not a dynamically bound variable. RQR triples stay fixed
throughout navigation/selection; coverage and selection progress change.

NER precedes triple extraction for corpus OpenIE and for RQR preparation.
OpenIE parsing includes source JSON-recovery/retry handling. An empty/failed
question extraction retains the source raw-question fallback. Such JSON
parsing recovery is not graph repair.

QA uses `openai/gpt-oss-120b`, temperature 0, low reasoning effort, and at most
1,024 completion tokens. Provider-specific reasoning accounting can consume
part of that completion budget; it is not 1,024 guaranteed answer tokens.
The prompt requests only short answer JSON, but permits inference from clues
and general knowledge when evidence is incomplete. It is not a strict
extractive, evidence-only reader. Top-5 is separate from retrieval Recall@10
or Recall@200 reporting.
