"""Command line interface."""

from __future__ import annotations

import argparse
import json
import sys

from lineageevo_eval.engine import EvaluationOptions, TOPK_CHOICES, SELECTION_METRICS, evaluate_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lineageevo-eval")
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate = subparsers.add_parser("evaluate", help="evaluate factor expressions with a LineageEvo-compatible protocol")
    evaluate.add_argument("--config", default="configs/eval.local.toml", help="path to eval.local.toml")
    evaluate.add_argument("--dataset", default=None, help="dataset key in config, for example csi500 or sp500")
    evaluate.add_argument("--input", required=True, help="JSONL file containing factor expressions")
    evaluate.add_argument("--select", choices=sorted(TOPK_CHOICES), default=None, help="topK selection mode")
    evaluate.add_argument("--selection-metric", choices=sorted(SELECTION_METRICS), default=None, help="valid_ic or train_ic")
    evaluate.add_argument("--output-dir", default=None, help="output directory; default creates a timestamped run under output_root")
    evaluate.add_argument("--no-backtest", action="store_true", help="skip Qlib backtest and only compute IC metrics")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "evaluate":
        try:
            result = evaluate_file(
                args.input,
                EvaluationOptions(
                    config_path=args.config,
                    dataset=args.dataset,
                    select=args.select,
                    selection_metric=args.selection_metric,
                    output_dir=args.output_dir,
                    run_backtest=False if args.no_backtest else None,
                ),
            )
        except Exception as exc:
            print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(result.summary, ensure_ascii=False, indent=2))
        return 0
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
