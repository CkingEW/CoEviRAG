# CoEviRAG Reproducibility Material

## Split Membership

`split_membership.json` contains the ordered public question identifiers for
1,000 training, 100 validation, and 1,000 test questions in each collection.
Training and validation use released training data; testing uses released
development data. Identifiers and normalized question strings are disjoint
across the three splits within each collection. SHA-256 fingerprints identify the processed split
files and fixed corpus used in the project. Question order is retained for
comparisons that use the first 100 test questions.

2WikiMultihopQA, HotpotQA, and MuSiQue use the original benchmark identifiers.
NQ and TriviaQA use the MRQA 2019 NaturalQuestionsShort and TriviaQA-web
releases. Each collection has its own deduplicated fixed corpus, formed from
the supplied contexts of its three splits. Available multi-hop distractors
are retained. The two MRQA adaptations preserve full supplied contexts and
use one answer-containing context per question as the labeled support;
they do not use answer-centered cropping. The manuscript describes the
support-label and corpus protocol.

## Prompt Templates

The four files in `prompts/` preserve the message strings and demonstrations:

- `ner.py`: named-entity extraction from a passage or question, with one example.
- `triple_extraction.py`: triples conditioned on a passage and its entity list.
- `query_triple_extraction.py`: fixed relational query patterns conditioned on
  the question and its extracted entities, with two examples; `?` marks an
  unspecified endpoint.
- `answer_generation.py`: the resolved answer-generation messages used by
  the GPT-OSS template, including the short JSON answer format.

OpenIE first extracts entities, then triples. RQR construction uses the same
entity-extraction messages, followed by the query-specific triple messages.
The final user message substitutes the full passage or question for
`${passage}` and a JSON object with a `named_entities` list for
`${named_entity_json}`. Answer generation substitutes the ranked passage
context and original question for `${context}` and `${question}`.
The source fragments document messages; relative imports refer to the
entity example and the conversion of `{field}` placeholders to `${field}`.
The answer-generation file resolves an implementation alias; all reported
main answers use GPT-OSS-120B, regardless of the backing module's filename.

`sha256.json` fingerprints every included material file. This archive
contains public benchmark identifiers and message definitions, not API
credentials, private paths, author identities, or new experimental results.
