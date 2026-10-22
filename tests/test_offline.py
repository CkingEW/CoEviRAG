"""Synthetic CPU tests. No dataset indexing, retrieval experiment, or API calls."""
import ast
import hashlib
import importlib
import json
from pathlib import Path
import socket
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import numpy as np
import torch

from prepare_data import clean_nq, normalize_mrqa, normalize_multihop, recover, write_json


class DataTests(unittest.TestCase):
    def test_nq_clean_context_matches_source_rule(self):
        self.assertEqual(clean_nq("<P> A  B </P>\n<P>C</P>"), "A B C")
        self.assertEqual(clean_nq("[PAR] A <p>B</p>"), "[PAR] A <p>B</p>")

    def test_source_json_serialization(self):
        value = [{"id": "q", "reference": ["caf\u00e9"]}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "split.json"
            write_json(path, value)
            # split_dataset.atomic_json_dump and the formal/MRQA writers use this form.
            expected = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
            self.assertEqual(path.read_bytes(), expected)

    def test_multihop_normalization_and_distractors(self):
        row = dict(_id="q", question="Q?", answer="A", answer_aliases=["A", "B"],
                   supporting_facts=[["T", 0]], context=[["T", [" One.", " Two. "]], ["D", ["Distractor."]]])
        value = normalize_multihop("hotpotqa", row)
        self.assertEqual(value["reference"], ["A", "B"])
        self.assertEqual(value["contexts"][0]["text"], "One. Two.")
        self.assertFalse(value["contexts"][1]["is_supporting"])

    def test_mrqa_full_context_and_title(self):
        raw = "<P>" + "A sufficiently long context containing the answer Paris. " * 2 + "</P>"
        start = raw.index("Paris")
        qa = dict(qid="q", question="Where?", answers=["Paris"],
                  detected_answers=[dict(char_spans=[[start, start + 4]])])
        value = normalize_mrqa("nq", dict(context=raw), qa, "train")
        self.assertEqual(value["contexts"][0]["text"], clean_nq(raw))
        self.assertEqual(value["contexts"][0]["title"], "NQ passage " + hashlib.sha256(clean_nq(raw).encode()).hexdigest()[:16])
        self.assertEqual(value["source"], "MRQA NaturalQuestionsShort train")

    def fixture(self, root, duplicate=False):
        def row(qid, question):
            return dict(_id=qid, question=question, answer="Answer", supporting_facts=[[qid, 0]],
                        context=[[qid, ["Some evidence."]]])
        train, dev, manifest = [root / name for name in ("train_raw.json", "dev_raw.json", "manifest.json")]
        write_json(train, [row("v", "  SAME   Question " if duplicate else "Validation?"), row("t", "same question")])
        write_json(dev, [row("e", "Test?")])
        splits = {name: dict(question_ids=[qid], processed_file_sha256="0" * 64)
                  for name, qid in (("train", "t"), ("val", "v"), ("test", "e"))}
        write_json(manifest, {"HotpotQA": dict(splits=splits, processed_corpus_sha256="0" * 64)})
        return train, dev, manifest

    def test_ordered_recovery_and_honest_byte_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train, dev, manifest = self.fixture(root)
            result = recover("hotpotqa", train, dev, root / "out", manifest)
            self.assertTrue(result["ordered_ids_recovered"])
            self.assertFalse(result["all_byte_fingerprints_match"])
            self.assertEqual(json.loads((root / "out/train.json").read_text())[0]["id"], "t")
            corpus = json.loads((root / "out/hotpotqa_corpus.json").read_text())
            self.assertEqual([r["title"] for r in corpus], ["t", "v", "e"])

    def test_normalized_questions_disjoint_across_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train, dev, manifest = self.fixture(root, duplicate=True)
            with self.assertRaisesRegex(ValueError, "Normalized questions overlap"):
                recover("hotpotqa", train, dev, root / "out", manifest)
            self.assertFalse((root / "out").exists())

    def test_manifest_id_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train, dev, manifest = self.fixture(root)
            payload = json.loads(manifest.read_text())
            payload["HotpotQA"]["splits"]["val"]["question_ids"] = ["t"]
            write_json(manifest, payload)
            with self.assertRaisesRegex(ValueError, "Split IDs overlap"):
                recover("hotpotqa", train, dev, root / "out", manifest)


class ProtocolTests(unittest.TestCase):
    def test_public_coevirag_namespace(self):
        from coevirag import CoEviRAG
        from coevirag.CoEviRAG import CoEviRAG as Implementation
        self.assertIs(CoEviRAG, Implementation)
        self.assertEqual(CoEviRAG.__name__, "CoEviRAG")
        self.assertEqual(CoEviRAG.__module__, "coevirag.CoEviRAG")
        self.assertFalse((ROOT / "repairrag").exists())

    def test_fixed_query_plan_and_progress(self):
        from coevirag.query_plan_evidence import dynamic_query_plan_enabled, append_selection_progress
        with self.assertRaises(ValueError):
            dynamic_query_plan_enabled(dict(uses_dynamic_query_plan=True))
        result = append_selection_progress(np.zeros((3, 51), dtype=np.float32), 2, 4)
        self.assertEqual(result.shape, (3, 53))
        np.testing.assert_array_equal(result[:, -2:], [[2, 2]] * 3)

    def test_final_selector_rejection(self):
        from coevirag.release_protocol import validate_selector
        good = dict(input_dim=53, uses_selection_progress_features=True,
                    uses_query_plan_features=True, selector_mode="residual", unary_mode="dpr")
        validate_selector(good, final=True)
        with self.assertRaises(ValueError):
            validate_selector(dict(good, input_dim=51), final=True)

    def test_safety_cache_order_and_global_ids(self):
        import pandas as pd
        from build_dpr_safety_cache import build_cache
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            split, ranking, store, output = [root / name for name in ("split.json", "dense.json", "store.parquet", "safety.npz")]
            write_json(split, [dict(question="Q?")])
            write_json(ranking, dict(results=[dict(question="Q?", docs=["A", "B"], scores=[.9, .8])]))
            pd.DataFrame(dict(hash_id=["outside-graph-a", "outside-graph-b"], content=["A", "B"])).to_parquet(store)
            build_cache(ranking, split, store, output, 2)
            with np.load(output) as cache:
                self.assertEqual(cache["passage_ids"].tolist(), [["outside-graph-a", "outside-graph-b"]])
            write_json(ranking, dict(results=[dict(question="Wrong?", docs=["A", "B"], scores=[.9, .8])]))
            with self.assertRaisesRegex(ValueError, "question order"):
                build_cache(ranking, split, store, output, 2)

    def test_unary60_and_selector53_shapes(self):
        from export_mixed_backend_cache import aligned_features
        from train_set_aware_evidence_selector import conditional_features
        from coevirag.query_plan_evidence import query_plan_state_features, append_selection_progress
        components = [dict(passage="a", fact=1., paragraph=None, similarity=.8, agreement=0., content="A"),
                      dict(passage="b", fact=None, paragraph=.4, similarity=.6, agreement=0., content="B")]
        base, unary, ids, _ = aligned_features(components, ["c"], [.9], lambda pid: pid, 1)
        self.assertEqual(set(ids), {"a", "b", "c"})
        self.assertEqual(unary.shape, (3, 60))
        pair = conditional_features(base, base[:, 0], 0, passage_ids=ids, openie_metadata={})
        plan = query_plan_state_features(ids, {}, [], [0])
        result = append_selection_progress(np.concatenate((pair, plan), axis=1), 1, 3)
        self.assertEqual(result.shape, (3, 53))

    def test_anchor_prefix_and_independent_tail(self):
        from evaluate_iterative_evidence_selector import select_ranking
        class ZeroSelector(torch.nn.Module):
            def forward(self, values):
                return torch.zeros((len(values), 1))
        base = np.zeros((5, 14), dtype=np.float32)
        scores = np.array([5., 4., 3., 2., 1.], dtype=np.float32)
        checkpoint = dict(input_dim=53, feature_mean=torch.zeros(53), feature_std=torch.ones(53),
                          selector_mode="residual", uses_query_plan_features=True,
                          uses_selection_progress_features=True, bridge_candidate_limit=0)
        ranking = select_ranking(base, np.array(list("abcde")), scores, ZeroSelector(), checkpoint,
                                 "setmax", 3, {}, "cpu", tail_scores=-scores, raw_query_triples=[])
        np.testing.assert_array_equal(ranking, [0, 1, 2, 4, 3])

    def test_constructor_and_export_without_models(self):
        core = importlib.import_module("coevirag.CoEviRAG")
        from coevirag.utils.config_utils import BaseConfig
        from run_frozen_fusion_sweep import build_components
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(core, "_get_llm_class", side_effect=lambda c: SimpleNamespace(global_config=c)), \
                patch.object(core, "_get_embedding_model_class", return_value=lambda **kwargs: object()), \
                patch.object(core, "EmbeddingStore", return_value=MagicMock()), \
                patch.object(core.CoEviRAG, "initialize_graph", return_value=object()):
            config = BaseConfig(save_dir=tmp, execution_device="cpu", graph_variant="unified_v1", fact_traversal_mode="role_aware")
            rag = core.CoEviRAG(global_config=config)
            self.assertFalse(rag._should_trigger_repair("q", {}))
            self.assertEqual(rag._repair_broken_path(), {})
            self.assertFalse(hasattr(rag, "_generate_repair_triples"))
            rag.get_fact_scores = MagicMock(return_value=np.array([1.]))
            rag.rank_facts = MagicMock(return_value=([0], [("A", "r", "B")]))
            rag._get_query_embedding_tensor = MagicMock(return_value=torch.zeros((1, 4)))
            rag.dense_passage_retrieval = MagicMock(return_value=([0], [1.]))
            rag.run_rl_graph_search_rerank = MagicMock(return_value=({"a": {"score": 1.}}, torch.zeros((1, 4))))
            rag._gate_fact_seeds_for_paragraph = lambda rows, emb: rows
            rag.run_paragraph_graph_search = MagicMock(return_value={"b": {"score": .5}})
            rag._merge_agent_passage_info = MagicMock(return_value={"a": {}, "b": {}})
            rag._build_candidate_components = MagicMock(return_value=[dict(passage="a")])
            self.assertEqual(build_components(rag, "q"), [dict(passage="a")])
            self.assertEqual(rag.run_paragraph_graph_search.call_args.kwargs["seed_passages"][0][0], "a")
        for key in ("repair_enabled", "repair_apply_results"):
            with self.assertRaises(ValueError):
                BaseConfig(**{key: True})

    def test_published_prompt_snapshots(self):
        manifest = json.loads((ROOT / "reproducibility/sha256.json").read_text())
        for name, expected in manifest.items():
            path = ROOT / "reproducibility" / name
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected, name)
        from coevirag.prompts.prompt_template_manager import PromptTemplateManager
        manager = PromptTemplateManager()
        self.assertEqual(set(manager.list_template_names()), {"ner", "triple_extraction", "query_triple_extraction", "rag_qa_gptoss"})
        self.assertEqual(len(manager.render("ner", passage="Synthetic question?")), 4)
        rqr = manager.render("query_triple_extraction", passage="Q?", named_entity_json='{"named_entities": []}')
        self.assertEqual(len(rqr), 6)
        self.assertIn('"?"', rqr[2]["content"])
        self.assertEqual(len(manager.render("rag_qa_gptoss", context="Synthetic context", question="Q?")), 2)

    def test_all_stage_commands_parse_without_execution(self):
        from coevirag import cli as coevirag
        config = json.loads((ROOT / "configs/final.json").read_text())
        for stage in coevirag.STAGES:
            command = coevirag.stage_command(stage, config, "val")
            module_name = Path(command[0]).stem
            module = importlib.import_module(module_name)
            # Stop immediately after argparse, before any file/model/API work.
            class Parsed(Exception):
                pass
            original = __import__("argparse").ArgumentParser.parse_args
            def parse_only(parser, *args, **kwargs):
                original(parser, *args, **kwargs)
                raise Parsed()
            with patch.object(sys, "argv", command), patch("argparse.ArgumentParser.parse_args", parse_only):
                with self.assertRaises(Parsed, msg=stage):
                    module.main()


if __name__ == "__main__":
    with patch.object(socket.socket, "connect", side_effect=AssertionError("Offline tests forbid network")), \
            patch("transformers.AutoModel.from_pretrained", side_effect=AssertionError("Model loads forbidden")), \
            patch("transformers.AutoTokenizer.from_pretrained", side_effect=AssertionError("Model loads forbidden")), \
            patch("torch.cuda.is_available", return_value=False):
        unittest.main(verbosity=2)
