from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

from .config import ExperimentConfig, LinearityConfig, RefinementConfig
from .datasets import dataset_summary
from .linearity import run_linearity_analysis
from .metrics import metrics_from_csv, verify_manifest
from .providers import CodexCliProvider
from .refinement import run_residual_guided_refinement
from .runner import run_experiment
from .security import validate_private_output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lsr-co2",
        description=(
            "Reproduce and run local-subspace CO2 adsorption experiments."
        ),
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True
    )

    verify = subparsers.add_parser(
        "verify",
        help="verify hashes and metrics in a local artifact manifest",
    )
    verify.add_argument(
        "--manifest", required=True
    )

    evaluate = subparsers.add_parser(
        "evaluate",
        help="evaluate a prediction CSV",
    )
    evaluate.add_argument("predictions")
    evaluate.add_argument("--prediction-column", required=True)
    evaluate.add_argument("--target-column", default="true")

    inspect = subparsers.add_parser(
        "inspect-data",
        help="validate and summarize a dataset",
    )
    inspect.add_argument("--config", required=True)

    run = subparsers.add_parser(
        "run",
        help="run a live Codex-backed experiment",
    )
    run.add_argument("--config", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--model", default="gpt-5.4")
    run.add_argument("--reasoning-effort", default="xhigh")
    run.add_argument("--codex-executable")
    run.add_argument("--timeout", type=int, default=600)
    run.add_argument("--split-seed", type=int)
    run.add_argument("--limit", type=int)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--allow-fallback", action="store_true")
    run.add_argument("--save-raw", action="store_true")

    refine = subparsers.add_parser(
        "refine",
        help="generate and freeze rules from an inner diagnostic split",
    )
    refine.add_argument("--config", required=True)
    refine.add_argument("--output-dir", required=True)
    refine.add_argument("--diagnostic-input")
    refine.add_argument("--model", default="gpt-5.4")
    refine.add_argument("--reasoning-effort", default="xhigh")
    refine.add_argument("--codex-executable")
    refine.add_argument("--timeout", type=int, default=2400)
    refine.add_argument("--save-prompt", action="store_true")
    refine.add_argument("--save-raw", action="store_true")
    run.add_argument(
        "--rules-path",
        help="override the frozen rules path from the experiment config",
    )

    linearity = subparsers.add_parser(
        "analyze-linearity",
        help="reproduce the Biochar local-linearity analysis",
    )
    linearity.add_argument("--config", required=True)
    linearity.add_argument("--output-dir", required=True)
    linearity.add_argument("--no-figure", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "verify":
        failures = verify_manifest(_resolve_from_cwd(args.manifest))
        if failures:
            for failure in failures:
                print(f"FAIL: {failure}")
            raise SystemExit(1)
        print(
            "PASS: all artifact hashes and metrics match the manifest"
        )
        return

    if args.command == "evaluate":
        metrics = metrics_from_csv(
            args.predictions,
            args.prediction_column,
            args.target_column,
        )
        print(json.dumps(metrics.to_dict(), indent=2))
        return

    if args.command == "refine":
        output_dir = validate_private_output_path(args.output_dir)
        refinement = RefinementConfig.from_json(args.config)
        experiment = ExperimentConfig.from_json(refinement.experiment_config)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
        )
        provider = CodexCliProvider(
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            executable=args.codex_executable,
            timeout_seconds=args.timeout,
            output_min=experiment.output_min,
            output_max=experiment.output_max,
        )
        summary = run_residual_guided_refinement(
            refinement,
            provider,
            output_dir,
            model_name=args.model,
            diagnostic_input=args.diagnostic_input,
            save_prompt=args.save_prompt,
            save_raw=args.save_raw,
        )
        print(json.dumps(asdict(summary), indent=2))
        return

    if args.command == "analyze-linearity":
        linearity_config = LinearityConfig.from_json(args.config)
        run = run_linearity_analysis(
            linearity_config,
            args.output_dir,
            save_figure=not args.no_figure,
        )
        print(json.dumps(asdict(run.summary), indent=2))
        return

    config = ExperimentConfig.from_json(args.config)
    if args.command == "inspect-data":
        print(json.dumps(dataset_summary(config), indent=2))
        return

    if args.split_seed is not None:
        config = config.with_split_seed(args.split_seed)
    if args.rules_path is not None:
        config = config.with_rules_path(args.rules_path)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    provider = CodexCliProvider(
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        executable=args.codex_executable,
        timeout_seconds=args.timeout,
        output_min=config.output_min,
        output_max=config.output_max,
    )
    summary = run_experiment(
        config,
        provider,
        validate_private_output_path(args.output),
        model_name=args.model,
        resume=args.resume,
        allow_fallback=args.allow_fallback,
        limit=args.limit,
        save_raw=args.save_raw,
    )
    payload = {
        "experiment": summary.experiment,
        "output_path": summary.output_path,
        "completed": summary.completed,
        "fallback_count": summary.fallback_count,
        "input_tokens": summary.input_tokens,
        "output_tokens": summary.output_tokens,
        "metrics": summary.metrics.to_dict(),
    }
    print(json.dumps(payload, indent=2))


def _resolve_from_cwd(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path.cwd() / path
