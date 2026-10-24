"""Train the unary tail and initial evidence selector for one dataset."""
import argparse
import json
from pathlib import Path
import torch
from coevirag.ranking.data import assert_split_separation, load_bundle
from coevirag.ranking.training import train_stage
from coevirag.release_protocol import BENCHMARK_DATASETS as DATASETS


def main():
    parser = argparse.ArgumentParser(description="Train independent 60D unary and initial 51D selector")
    parser.add_argument("--output_root", type=Path, default=Path("outputs"))
    parser.add_argument("--root", type=Path, help="Override the dataset's models/rankers directory")
    parser.add_argument("--datasets", choices=(*DATASETS, "sample"), nargs=1, required=True)
    parser.add_argument("--tail_epochs", type=int, default=30)
    parser.add_argument("--selector_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.root is None:
        args.root = args.output_root / args.datasets[0] / "models/rankers"
    if args.batch_size < 1 or min(args.tail_epochs, args.selector_epochs) < 0:
        parser.error("Invalid batch size or epoch count")
    assert_split_separation(args.datasets)
    args.root.mkdir(parents=True, exist_ok=True)
    if (args.root / "tail_history.json").exists():
        parser.error("Choose a new backend directory")
    torch.set_num_threads(4)
    bundles = {d: load_bundle(d, args.output_root, 0, 0) for d in args.datasets}
    tail = train_stage(bundles, args.root, args.tail_epochs, args.batch_size, args.seed)
    selector = train_stage(bundles, args.root, args.selector_epochs, args.batch_size,
                           args.seed, selector=True, tail_ckpt=tail)
    (args.root / "training_summary.json").write_text(json.dumps(dict(
        datasets=args.datasets, tail_best_epoch=tail["epoch"],
        selector_best_epoch=selector["epoch"], agents="frozen", repair=False), indent=2))


if __name__ == "__main__":
    main()
