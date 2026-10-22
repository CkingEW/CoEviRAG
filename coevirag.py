"""Run the CLI without importing the model runtime during --dry-run."""
from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "coevirag" / "cli.py"),
                   run_name="__main__")
