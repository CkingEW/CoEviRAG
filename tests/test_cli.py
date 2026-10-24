"""Standard-library workflow tests; no model imports, subprocesses, or APIs."""
from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("coevirag_cli", ROOT / "coevirag/cli.py")
cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cli)


class WorkflowTests(unittest.TestCase):
    def invoke(self, *arguments):
        output = io.StringIO()
        with patch.object(sys, "argv", ["coevirag.py", *arguments]), \
                redirect_stdout(output), redirect_stderr(io.StringIO()):
            cli.main()
        return output.getvalue()

    def test_training_order_and_no_test_questions(self):
        self.assertEqual(cli.workflow_steps("train"), [
            ("index", "train"), ("train-policies", "train"),
            ("queries", "train"), ("queries", "val"),
            ("export", "train"), ("export", "val"),
            ("dense", "val"), ("safety", "val"),
            ("train-backend", "val"), ("train-selector", "val"),
        ])

    def test_retrieval_order_and_optional_qa(self):
        names = ["queries", "export", "dense", "safety", "retrieve"]
        for split in ("val", "test"):
            self.assertEqual(cli.workflow_steps("retrieve", split), [(s, split) for s in names])
            self.assertEqual(cli.workflow_steps("retrieve", split, with_qa=True),
                             [(s, split) for s in [*names, "qa"]])
            self.assertEqual(cli.workflow_steps("retrieve", split, stage_only=True),
                             [("retrieve", split)])

    def test_individual_stages_remain_available(self):
        for stage in cli.STAGES:
            if stage != "retrieve":
                self.assertEqual(cli.workflow_steps(stage, "val"), [(stage, "val")])

    def test_invalid_workflow_options(self):
        invalid = [("train", "--split", "test"), ("train", "--with-qa"),
                   ("index", "--stage-only"), ("retrieve", "--split", "train"),
                   ("retrieve", "--stage-only", "--with-qa")]
        with patch.object(cli.subprocess, "run") as run:
            for arguments in invalid:
                with self.subTest(arguments=arguments), self.assertRaises(SystemExit) as error:
                    self.invoke(*arguments)
                self.assertEqual(error.exception.code, 2)
            run.assert_not_called()

    def test_dry_run_is_complete_and_never_executes(self):
        with patch.object(cli.subprocess, "run") as run, \
                patch.object(cli, "validate_workflow") as validate:
            for dataset in cli.DATASETS:
                for stage, options, count in (("train", [], 10), ("retrieve", [], 5),
                                              ("retrieve", ["--with-qa"], 6)):
                    plan = json.loads(self.invoke(stage, "--dataset", dataset, *options, "--dry-run"))
                    self.assertEqual(plan["dataset"], dataset)
                    self.assertEqual(len(plan["steps"]), count)
                    for step in plan["steps"]:
                        self.assertEqual(step["command"][0], sys.executable)
                        self.assertIn(dataset, json.dumps(step["command"]))
            run.assert_not_called()
            validate.assert_not_called()

    def test_atomic_dry_run_format_is_preserved(self):
        for arguments in (("index",), ("retrieve", "--stage-only")):
            result = json.loads(self.invoke(*arguments, "--dry-run"))
            self.assertIn("command", result)
            self.assertNotIn("steps", result)

    def test_sample_uses_small_corpus_only_for_sample(self):
        config = json.loads((ROOT / "configs/sample.json").read_text())
        self.assertEqual(config["dataset"], "sample")
        for key in ("policy_epochs", "tail_epochs", "initial_selector_epochs", "multistate_epochs"):
            self.assertEqual(config[key], 1)
        for stage in ("dense", "safety"):
            self.assertIn("--allow_small_corpus", cli.stage_command(stage, config, "val"))
            config["dataset"] = "hotpotqa"
            self.assertNotIn("--allow_small_corpus", cli.stage_command(stage, config, "val"))
            config["dataset"] = "sample"

    def test_dataset_sample_selects_one_epoch_configuration(self):
        for stage, options in (("train", []), ("retrieve", ["--with-qa"])):
            automatic = json.loads(self.invoke(stage, "--dataset", "sample", *options, "--dry-run"))
            explicit = json.loads(self.invoke(stage, "--config", str(ROOT / "configs/sample.json"),
                                             *options, "--dry-run"))
            self.assertEqual(automatic, explicit)
        plan = json.loads(self.invoke("train", "--dataset", "sample", "--dry-run"))
        training = plan["steps"][1]["command"]
        self.assertEqual(training[training.index("--epochs") + 1], "1")

    def test_explicit_config_takes_precedence_over_sample_default(self):
        config = json.loads((ROOT / "configs/final.json").read_text())
        result = json.loads(self.invoke("train", "--dataset", "sample", "--config",
                                       str(ROOT / "configs/final.json"), "--dry-run"))
        training = result["steps"][1]["command"]
        self.assertEqual(training[training.index("--epochs") + 1], str(config["policy_epochs"]))

    def test_execution_order_and_default_gpu(self):
        with patch.object(cli, "validate_workflow"), \
                patch.dict(cli.os.environ, {}, clear=True), \
                patch.object(cli.subprocess, "run") as run:
            self.invoke("train", "--dataset", "musique")
        self.assertEqual(len(run.call_args_list), 10)
        expected = cli.workflow_steps("train")
        config = json.loads((ROOT / "configs/final.json").read_text())
        config["dataset"] = "musique"
        for call, (stage, split) in zip(run.call_args_list, expected):
            self.assertEqual(call.args[0], [sys.executable, *cli.stage_command(stage, config, split)])
            self.assertEqual(call.kwargs["cwd"], ROOT)
            self.assertTrue(call.kwargs["check"])
            self.assertEqual(call.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "1")

    def test_explicit_gpu_and_qa_are_preserved(self):
        with patch.object(cli, "validate_workflow"), \
                patch.dict(cli.os.environ, {"CUDA_VISIBLE_DEVICES": "0"}), \
                patch.object(cli.subprocess, "run") as run:
            self.invoke("retrieve", "--with-qa", "--split", "val")
        self.assertEqual(run.call_count, 6)
        self.assertEqual(run.call_args.args[0][1], "scripts/generate_answers.py")
        for call in run.call_args_list:
            self.assertEqual(call.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "0")

    def test_failure_stops_later_stages(self):
        failure = subprocess.CalledProcessError(1, "synthetic-stage")
        with patch.object(cli, "validate_workflow"), \
                patch.object(cli.subprocess, "run", side_effect=[None, failure]) as run:
            with self.assertRaises(subprocess.CalledProcessError):
                self.invoke("retrieve", "--with-qa")
        self.assertEqual(run.call_count, 2)

    def test_missing_checkpoints_fail_before_execution(self):
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(cli, "ROOT", Path(temporary)), \
                patch.object(cli.subprocess, "run") as run:
            with self.assertRaises(SystemExit) as error:
                self.invoke("retrieve", "--config", str(ROOT / "configs/final.json"))
            self.assertEqual(error.exception.code, 2)
            run.assert_not_called()

    def test_training_history_protection(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(cli, "ROOT", Path(temporary)):
            config = {"dataset": "hotpotqa"}
            cli.validate_workflow("train", config)
            for path in ("outputs/hotpotqa/models/rankers/tail_history.json",
                         "outputs/hotpotqa/models/selector/selector_history.json"):
                history = Path(temporary) / path
                history.parent.mkdir(parents=True, exist_ok=True)
                history.write_text("[]")
                with self.assertRaisesRegex(ValueError, "already exist"):
                    cli.validate_workflow("train", config)
                history.unlink()

    def test_complete_checkpoints_pass_preflight(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(cli, "ROOT", Path(temporary)):
            files = ["outputs/hotpotqa/models/agents/best_policy.pth",
                     "outputs/hotpotqa/models/agents/best_paragraph_policy.pth",
                     "outputs/hotpotqa/models/rankers/independent_tail_seed42.pth",
                     "outputs/hotpotqa/models/selector/multistate_selector_seed42.pth"]
            for name in files:
                path = Path(temporary) / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"synthetic checkpoint placeholder")
            cli.validate_workflow("retrieve", {"dataset": "hotpotqa"})

    def test_all_stage_outputs_are_dataset_scoped(self):
        config = json.loads((ROOT / "configs/final.json").read_text())
        for dataset in cli.DATASETS:
            config["dataset"] = dataset
            for stage in cli.STAGES:
                command = cli.stage_command(stage, config, "test")
                for value in command:
                    if value.startswith("outputs/"):
                        self.assertTrue(value.startswith(f"outputs/{dataset}/"), (stage, value))
                if stage in ("queries", "export", "retrieve", "qa"):
                    self.assertIn(f"outputs/{dataset}/models/agents" if stage in ("queries", "export")
                                  else f"outputs/{dataset}/results/{'qa/test_top5.json' if stage == 'qa' else 'retrieval/test.json'}",
                                  command)


if __name__ == "__main__":
    unittest.main(verbosity=2)
