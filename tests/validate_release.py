"""AST/compile, runtime imports, CLI help, and non-disclosing release scan."""
import argparse
import ast
from contextlib import redirect_stderr, redirect_stdout
import importlib
import importlib.util
import io
import json
from pathlib import Path
import re
import socket
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]


def scan():
    patterns = [
        re.compile(r"sk-(?:or-v1-|proj-)?[A-Za-z0-9_-]{20,}"),
        re.compile(r"(?:ghp_|github_pat_)[A-Za-z0-9_]{20,}"),
        re.compile(r"nvapi-[A-Za-z0-9_-]{16,}"),
        re.compile(r"hf_[A-Za-z0-9]{16,}"),
        re.compile(r"AKIA[A-Z0-9]{16}"),
        re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
        re.compile(r"/(?:home|root|Users)/[A-Za-z0-9_.-]+/"),
    ]
    findings = 0
    files = [path for path in ROOT.rglob("*") if path.is_file()]
    for path in files:
        if path.is_symlink():
            findings += 1
            continue
        if path.name == ".env" or path.name.startswith("API") or "credential" in path.name.lower():
            findings += 1
            continue
        if any(part in {".git", "__pycache__", "outputs", "logs", "skills"} for part in path.relative_to(ROOT).parts):
            findings += 1
        text = path.read_text(encoding="utf-8")
        findings += sum(bool(pattern.search(text)) for pattern in patterns)
    if findings:
        raise AssertionError(f"Release scan found {findings} issues; matched content is not printed")
    return dict(files=len(files), bytes=sum(path.stat().st_size for path in files), findings=0)


def main():
    files = sorted(ROOT.rglob("*.py"))
    for path in files:
        content = path.read_text(encoding="utf-8")
        ast.parse(content, filename=str(path.relative_to(ROOT)))
        compile(content, str(path.relative_to(ROOT)), "exec")
        if path.is_relative_to(ROOT / "coevirag") or path.parent == ROOT / "scripts":
            for node in ast.walk(ast.parse(content)):
                modules = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                           else [node.module] if isinstance(node, ast.ImportFrom) and not node.level else [])
                for module in modules:
                    if module and importlib.util.find_spec(module) is None:
                        raise AssertionError("Unresolved import dependency: " + module)
    runtime = []
    for path in sorted((ROOT / "coevirag").rglob("*.py")):
        parts = list(path.relative_to(ROOT).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        importlib.import_module(".".join(parts))
        runtime.append(".".join(parts))
    helps = []
    for path in sorted((ROOT / "scripts").glob("*.py")):
        module = importlib.import_module(path.stem)
        runtime.append(path.stem)
        if callable(getattr(module, "main", None)):
            with patch.object(sys, "argv", [str(path.relative_to(ROOT)), "--help"]), \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                try:
                    module.main()
                except SystemExit as error:
                    if error.code != 0:
                        raise
                else:
                    raise AssertionError("CLI help must stop at argparse")
            helps.append(path.stem)
    from coevirag import cli as coevirag
    with patch.object(sys, "argv", ["coevirag.py", "--help"]), redirect_stdout(io.StringIO()):
        try:
            coevirag.main()
        except SystemExit as error:
            if error.code != 0:
                raise
    print(json.dumps(dict(ast_compiled=len(files), runtime_imports=len(runtime),
                         cli_help=len(helps) + 1, scan=scan()), indent=2))


if __name__ == "__main__":
    import torch
    import transformers
    with patch.object(socket.socket, "connect", side_effect=AssertionError("Network forbidden")), \
            patch.object(torch, "load", side_effect=AssertionError("Checkpoint loading forbidden")), \
            patch.object(transformers.AutoModel, "from_pretrained", side_effect=AssertionError("Model loading forbidden")), \
            patch.object(transformers.AutoTokenizer, "from_pretrained", side_effect=AssertionError("Model loading forbidden")):
        main()
