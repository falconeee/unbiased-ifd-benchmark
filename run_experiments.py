#!/usr/bin/env python3
"""Run the benchmark experiments from the command line and save one JSON per experiment.

This is the script form of the v0 notebooks kept in ``v0_experiments/``: same datasets, same
unbiased folds, same nine architectures, same hyperparameters. Results land in
``results/<suite>/<DATASET>/<experiment>.json`` using the schema documented in
``README.md``.

Setup (recommended: uv, from the repository root)
-------------------------------------------------
    uv venv --no-project
    uv pip install -r requirements.txt

``uv run --no-project`` then uses ``./.venv`` without activating anything. Drop the prefix if
you activated the environment yourself.

Examples
--------
    # everything, resumable, in the background
    nohup uv run --no-project python run_experiments.py \
        --all --resume --keep-going > run.out 2>&1 &

    # one notebook
    uv run --no-project python run_experiments.py --suite single_round --dataset CWRU12k

    # one experiment, quick smoke test
    uv run --no-project python run_experiments.py --suite single_round --dataset UOC \
        --model cnn1d --epochs 2 --max-rounds 1

    # what would run
    uv run --no-project python run_experiments.py --all --list
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import traceback
import warnings
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Dict, List, Optional

# The notebooks open with ``warnings.filterwarnings("ignore")``; without it sklearn floods
# the log with UndefinedMetricWarning on folds where a class is never predicted.
warnings.filterwarnings("ignore")

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
# Make ``src`` importable no matter where the script is invoked from.
sys.path.insert(0, REPO_ROOT)


# --------------------------------------------------------------------------- logging
class Tee:
    """Mirror stdout/stderr into a log file so background runs keep a full transcript."""

    def __init__(self, stream, path):
        self.stream = stream
        self.file = open(path, "a", buffering=1, encoding="utf-8")

    def write(self, data):
        self.stream.write(data)
        self.file.write(data)

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def log(message: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


# --------------------------------------------------------------------------- device
def available_devices() -> "OrderedDict[str, str]":
    """Backends torch can actually use here, in descending order of preference.

    ``cpu`` is always present, so the mapping is never empty.
    """
    import torch

    found: "OrderedDict[str, str]" = OrderedDict()
    if torch.cuda.is_available():
        found["cuda"] = f"{torch.cuda.get_device_name(0)} (x{torch.cuda.device_count()})"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        found["mps"] = "Apple Silicon GPU (Metal Performance Shaders)"
    found["cpu"] = platform.processor() or platform.machine() or "cpu"
    return found


def resolve_device(requested: Optional[str]) -> str:
    """Return the torch device string to train on.

    Without ``--device``, picks the fastest backend available: ``cuda``, then ``mps``, then
    ``cpu``. With ``--device``, validates the request and fails immediately rather than
    letting the run die inside the first fold.
    """
    import torch

    found = available_devices()

    if not requested:
        chosen = next(iter(found))
        others = ", ".join(k for k in found if k != chosen) or "none"
        log(f"device: {chosen} auto-detected -- {found[chosen]} (also available: {others})")
        if chosen == "mps":
            log("      MPS note: results may differ marginally from CPU/CUDA (different "
                "kernels). Set PYTORCH_ENABLE_MPS_FALLBACK=1 before launching if an "
                "unsupported op aborts a run.")
        elif chosen == "cpu":
            log("      no GPU backend found; a full run on CPU takes considerably longer")
        return chosen

    kind = requested.split(":")[0]
    if kind not in found:
        raise SystemExit(
            f"--device {requested!r}: torch reports no usable {kind!r} backend on this "
            f"machine. Available: {', '.join(found)}."
        )
    if kind == "cuda" and ":" in requested:
        index = int(requested.split(":", 1)[1])
        if index >= torch.cuda.device_count():
            raise SystemExit(
                f"--device {requested!r}: only {torch.cuda.device_count()} CUDA device(s) "
                f"visible (valid indices 0..{torch.cuda.device_count() - 1})."
            )
    log(f"device: {requested} requested -- {found[kind]}")
    return requested


# ------------------------------------------------------------------------------ CLI
def build_parser() -> argparse.ArgumentParser:
    from src.registry import DATASET_NAMES, MODEL_KEYS, SUITES

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sel = parser.add_argument_group("selection")
    sel.add_argument("--all", action="store_true",
                     help="run every suite, dataset and model (same as passing no filter)")
    sel.add_argument("--suite", action="append", choices=SUITES,
                     help="restrict to a suite (repeatable)")
    sel.add_argument("--dataset", action="append", choices=DATASET_NAMES,
                     help="restrict to a dataset (repeatable)")
    sel.add_argument("--model", "--experiment", dest="model", action="append", choices=MODEL_KEYS,
                     help="restrict to a model key (repeatable)")
    sel.add_argument("--list", action="store_true", help="print the selection and exit")

    paths = parser.add_argument_group("paths")
    paths.add_argument("--data-root", default=os.path.join(REPO_ROOT, "data"),
                       help="where raw/converted datasets and fold caches live (default: %(default)s)")
    paths.add_argument("--output-dir", default=os.path.join(REPO_ROOT, "results"),
                       help="where the result JSONs are written (default: %(default)s)")
    paths.add_argument("--artifacts-dir", default=None,
                       help="where per-fold loss curves and checkpoints go "
                            "(default: <output-dir>/_artifacts)")
    paths.add_argument("--log-file", default=None,
                       help="append the full transcript to this file "
                            "(default: <output-dir>/run.log)")
    paths.add_argument("--no-download", action="store_true",
                       help="do not download raw datasets (they must already be present)")

    run = parser.add_argument_group("execution")
    run.add_argument("--device", default=None,
                     help="torch device: cuda, cuda:N, mps or cpu "
                          "(default: auto-detect, preferring cuda, then mps, then cpu)")
    run.add_argument("--seed", type=int, default=42, help="random seed (default: %(default)s)")
    run.add_argument("--resume", action="store_true",
                     help="skip experiments whose result JSON already exists")
    run.add_argument("--keep-going", action="store_true",
                     help="carry on to the next experiment when one raises")
    run.add_argument("--no-artifacts", action="store_true",
                     help="do not write loss curves and model checkpoints")
    run.add_argument("--folds-source", choices=("notebook", "generate"), default="notebook",
                     help="multiround folds: rebuild from the notebooks' printed design "
                          "(default) or run the full combination search, which needs ~15 GB "
                          "of RAM for CWRU12k/PU")

    override = parser.add_argument_group("overrides (for smoke tests)")
    override.add_argument("--epochs", type=int, default=None, help="override num_epochs")
    override.add_argument("--pretrain-epochs", type=int, default=None,
                          help="override pretrain_epochs")
    override.add_argument("--batch-size", type=int, default=None, help="override batch_size")
    override.add_argument("--max-rounds", type=int, default=None,
                          help="run at most this many rounds of a multiround experiment")
    return parser


# ------------------------------------------------------------------------ execution
def experiment_configuration(spec, input_length: int, num_classes: int, args) -> OrderedDict:
    cfg = OrderedDict()
    cfg["batch_size"] = args.batch_size if args.batch_size is not None else spec.batch_size
    cfg["lr"] = spec.lr
    cfg["num_epochs"] = args.epochs if args.epochs is not None else spec.num_epochs
    cfg["pretrain_epochs"] = (
        args.pretrain_epochs if args.pretrain_epochs is not None else spec.pretrain_epochs
    )
    if spec.recon_loss_weight is not None:
        cfg["recon_loss_weight"] = spec.recon_loss_weight
    if spec.sparsity_target is not None:
        cfg["sparsity_target"] = spec.sparsity_target
    if spec.sparsity_weight is not None:
        cfg["sparsity_weight"] = spec.sparsity_weight
    cfg["input_length"] = input_length
    cfg["num_classes"] = num_classes
    return cfg


def run_one_round(spec, cfg, fold_idxs, X, y, input_length, num_classes, args,
                  artifacts_dir: str, round_label: str):
    """Build the model and run one full cross validation; returns (results, experiment)."""
    import torch.nn as nn

    from src.experiment import DeepLearningExperiment, seed_everything
    from src.models import build_model

    # The notebooks deep-copy one prototype for every round, so each round starts from the
    # same initial weights; re-seeding before building reproduces that deterministically.
    seed_everything(args.seed)
    model = build_model(spec.key, input_length=input_length, num_classes=num_classes)

    kwargs = dict(
        name=f"{spec.experiment_name}{round_label}",
        description=spec.description,
        data_fold_idxs=fold_idxs,
        model=model,
        batch_size=cfg["batch_size"],
        lr=cfg["lr"],
        num_epochs=cfg["num_epochs"],
        pretrain_epochs=cfg["pretrain_epochs"],
        output_dir=artifacts_dir,
        device=args.device,
        X=X,
        y=y,
        seed=args.seed,
        save_artifacts=not args.no_artifacts,
    )
    if spec.is_autoencoder:
        kwargs["criterion"] = nn.CrossEntropyLoss()
        kwargs["reconstruction_criterion"] = nn.MSELoss()
        kwargs["recon_loss_weight"] = spec.recon_loss_weight if spec.recon_loss_weight else 1.0
        if spec.sparsity_target is not None:
            kwargs["sparsity_target"] = spec.sparsity_target
            kwargs["sparsity_weight"] = spec.sparsity_weight or 0.0

    experiment = DeepLearningExperiment(**kwargs)
    results = experiment.run()
    return results, experiment


def run_experiment(spec, context, args) -> OrderedDict:
    """Run one experiment (all rounds) and return the document to serialise."""
    from src import serialize

    X, y = context["X"], context["y"]
    input_length, num_classes = context["input_length"], context["num_classes"]
    rounds_folds = context["rounds"]
    cfg = experiment_configuration(spec, input_length, num_classes, args)

    artifacts_dir = os.path.join(args.artifacts_dir, spec.suite, spec.dataset, spec.slug)
    started = time.time()

    if spec.protocol == "single_round":
        results, experiment = run_one_round(
            spec, cfg, rounds_folds[0], X, y, input_length, num_classes, args,
            artifacts_dir, round_label="",
        )
        results_block = serialize.single_round_results(
            results, experiment.history, spec.training_stage, cfg["num_epochs"],
            experiment.fold_errors, time.time() - started,
        )
    else:
        total_rounds = len(rounds_folds)
        planned = total_rounds
        if args.max_rounds is not None:
            total_rounds = min(total_rounds, args.max_rounds)
        round_blocks = []
        for round_idx in range(total_rounds):
            log(f"    round {round_idx + 1}/{total_rounds}")
            round_started = time.time()
            results, experiment = run_one_round(
                spec, cfg, rounds_folds[round_idx], X, y, input_length, num_classes, args,
                artifacts_dir, round_label=f"_round_{round_idx}",
            )
            round_blocks.append(
                serialize.round_block(
                    round_idx + 1, total_rounds, results, experiment.history,
                    spec.training_stage, cfg["num_epochs"], experiment.fold_errors,
                    time.time() - round_started,
                )
            )
        results_block = serialize.multi_round_results(
            round_blocks, planned, time.time() - started
        )

    run_info = OrderedDict(
        started_at=context["run_started_at"],
        finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        duration_seconds=round(time.time() - started, 3),
        seed=args.seed,
        device=context["device"],
        folds_source=context["folds_source"],
        overrides=context["overrides"],
        environment=context["environment"],
    )

    return serialize.experiment_document(
        spec=spec.as_dict(),
        dataset=spec.dataset,
        suite=spec.suite,
        protocol=spec.protocol,
        model_class=spec.model_class,
        run_info=run_info,
        configuration=cfg,
        results=results_block,
        fold_design=context.get("fold_design"),
        notes=context.get("notes"),
    )


def notebook_context(suite: str, dataset: str, specs, args) -> Dict:
    """Load the dataset and folds once per notebook, shared by its nine experiments."""
    import numpy as np

    from src import folds as folds_mod
    from src.datasets import DATASETS, load_deep_dataset
    from src.experiment import load_xy

    spec0 = specs[0]
    protocol = spec0.protocol
    dataset_spec = DATASETS[dataset]

    log(f"  loading dataset {dataset} ({protocol})")
    deep_dataset = load_deep_dataset(dataset_spec, args.data_root, download=not args.no_download)
    X, y, _ = load_xy(deep_dataset)
    input_length = int(X.shape[-1])
    num_classes = int(len(np.unique(y)))
    log(f"  {len(X)} samples, input_length={input_length}, num_classes={num_classes}")

    rounds, folds_source = folds_mod.build_folds(
        dataset_spec, deep_dataset, protocol, args.data_root, args.folds_source
    )
    log(f"  folds: {len(rounds)} round(s), source={folds_source}")

    context = {
        "X": X,
        "y": y,
        "input_length": input_length,
        "num_classes": num_classes,
        "rounds": rounds,
        "folds_source": folds_source,
    }
    if protocol == "multi_round":
        designs = folds_mod.load_fold_designs()
        if dataset in designs:
            context["fold_design"] = designs[dataset]
    if protocol != suite:
        context["notes"] = (
            f"The cells live in '{suite}_experiments', but the notebook runs a {protocol} "
            f"experiment (it uses 'folds_singleround_deep'). Reproduced as-is."
        )
    return context


# ----------------------------------------------------------------------- reporting
def rebuild_indices(output_dir: str) -> int:
    """Regenerate ``_index.json`` per notebook and ``index.json`` at the root."""
    import glob

    from src import serialize

    all_rows = []
    for suite_dir in sorted(glob.glob(os.path.join(output_dir, "*"))):
        if not os.path.isdir(suite_dir):
            continue
        suite = os.path.basename(suite_dir)
        if suite.startswith("_"):
            continue  # _artifacts and friends are not result folders
        for ds_dir in sorted(glob.glob(os.path.join(suite_dir, "*"))):
            if not os.path.isdir(ds_dir):
                continue
            dataset = os.path.basename(ds_dir)
            rows = []
            for path in sorted(glob.glob(os.path.join(ds_dir, "*.json"))):
                if os.path.basename(path) == "_index.json":
                    continue
                with open(path, encoding="utf-8") as fh:
                    doc = json.load(fh)
                summary = doc.get("results", {}).get("summary", {})
                rows.append(OrderedDict(
                    file=os.path.basename(path),
                    experiment_name=doc.get("experiment_name"),
                    model=doc.get("model"),
                    protocol=doc.get("protocol"),
                    status=doc.get("status"),
                    mean_accuracy=summary.get("mean_accuracy"),
                    std_accuracy=summary.get("std_accuracy"),
                    mean_f1_score=summary.get("mean_f1_score"),
                    std_f1_score=summary.get("std_f1_score"),
                ))
                all_rows.append(OrderedDict(suite=suite, dataset=dataset, **rows[-1]))
            if rows:
                serialize.write_json(
                    os.path.join(ds_dir, "_index.json"),
                    OrderedDict(dataset=dataset, suite=suite,
                                source_notebook=f"v0_experiments/{suite}_experiments/SignalAI_Framework_{dataset}.ipynb",
                                num_experiments=len(rows), experiments=rows),
                )
    if all_rows:
        serialize.write_json(
            os.path.join(output_dir, "index.json"),
            OrderedDict(total_experiments=len(all_rows), experiments=all_rows),
        )
    return len(all_rows)


def environment_info(device: str) -> OrderedDict:
    import sklearn
    import torch

    info = OrderedDict()
    info["python"] = platform.python_version()
    info["platform"] = platform.platform()
    info["torch"] = torch.__version__
    info["sklearn"] = sklearn.__version__
    for package in ("vibdata", "signalAI"):
        try:
            from importlib.metadata import version

            info[package] = version(package)
        except Exception:  # noqa: BLE001
            info[package] = "unknown"
    info["device"] = device
    info["devices_available"] = list(available_devices())
    if device.startswith("cuda") and torch.cuda.is_available():
        info["accelerator"] = torch.cuda.get_device_name(0)
        info["gpu"] = info["accelerator"]  # kept: earlier result files use this key
        info["cuda"] = torch.version.cuda
    elif device.startswith("mps"):
        info["accelerator"] = "Apple Silicon GPU (MPS)"
        info["gpu"] = info["accelerator"]
    return info


# ---------------------------------------------------------------------------- main
def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    from src.registry import DATASET_NAMES, MODEL_KEYS, SUITES, select

    suites = tuple(args.suite) if args.suite else SUITES
    datasets = tuple(args.dataset) if args.dataset else DATASET_NAMES
    keys = tuple(args.model) if args.model else MODEL_KEYS
    selection = list(select(suites, datasets, keys))

    if args.list:
        print(f"{len(selection)} experiment(s) selected\n")
        print(f"{'suite':<13}{'dataset':<9}{'model':<10}{'protocol':<13}experiment")
        for spec in selection:
            print(f"{spec.suite:<13}{spec.dataset:<9}{spec.key:<10}{spec.protocol:<13}"
                  f"{spec.experiment_name}")
        return 0

    if not selection:
        parser.error("no experiment matches the given filters")

    os.makedirs(args.output_dir, exist_ok=True)
    if args.artifacts_dir is None:
        args.artifacts_dir = os.path.join(args.output_dir, "_artifacts")
    if args.log_file is None:
        args.log_file = os.path.join(args.output_dir, "run.log")

    tee_out = Tee(sys.stdout, args.log_file)
    sys.stdout = tee_out
    sys.stderr = Tee(sys.stderr, args.log_file)

    from src import serialize

    device = resolve_device(args.device)
    args.device = device
    run_started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    environment = environment_info(device)
    overrides = OrderedDict(
        (name, getattr(args, name))
        for name in ("epochs", "pretrain_epochs", "batch_size", "max_rounds")
        if getattr(args, name) is not None
    )

    log(f"{len(selection)} experiment(s) selected | device={device} | seed={args.seed}")
    log(f"output-dir={args.output_dir}  data-root={args.data_root}")
    if overrides:
        log(f"overrides: {dict(overrides)}")

    done, skipped, failed = [], [], []
    by_notebook: "OrderedDict[str, list]" = OrderedDict()
    for spec in selection:
        by_notebook.setdefault(f"{spec.suite}/{spec.dataset}", []).append(spec)

    for notebook_key, specs in by_notebook.items():
        suite, dataset = notebook_key.split("/")
        target_dir = os.path.join(args.output_dir, suite, dataset)

        pending = specs
        if args.resume:
            pending = [s for s in specs if not os.path.exists(os.path.join(target_dir, s.slug + ".json"))]
            for spec in specs:
                if spec not in pending:
                    skipped.append(f"{notebook_key}/{spec.key}")
        if not pending:
            log(f"{notebook_key}: nothing to do (all results present)")
            continue

        log(f"=== {notebook_key} -- {len(pending)} experiment(s)")
        try:
            context = notebook_context(suite, dataset, pending, args)
        except Exception as exc:  # noqa: BLE001
            log(f"!! {notebook_key}: failed to prepare dataset/folds: {exc}")
            traceback.print_exc()
            failed.extend(f"{notebook_key}/{s.key}" for s in pending)
            if args.keep_going:
                continue
            return 1
        context["run_started_at"] = run_started_at
        context["device"] = device
        context["environment"] = environment
        context["overrides"] = overrides

        os.makedirs(target_dir, exist_ok=True)
        for spec in pending:
            label = f"{notebook_key}/{spec.key} ({spec.experiment_name})"
            log(f"--- {label}")
            try:
                document = run_experiment(spec, context, args)
            except Exception as exc:  # noqa: BLE001
                log(f"!! {label} failed: {exc}")
                traceback.print_exc()
                failed.append(f"{notebook_key}/{spec.key}")
                if args.keep_going:
                    continue
                return 1
            path = os.path.join(target_dir, spec.slug + ".json")
            serialize.write_json(path, document)
            summary = document["results"].get("summary", {})
            log(f"    saved {os.path.relpath(path, REPO_ROOT)} | "
                f"acc={summary.get('mean_accuracy')} f1={summary.get('mean_f1_score')}")
            done.append(f"{notebook_key}/{spec.key}")

    total = rebuild_indices(args.output_dir)
    manifest = OrderedDict(
        started_at=run_started_at,
        finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        command=" ".join(sys.argv),
        seed=args.seed,
        device=device,
        folds_source=args.folds_source,
        overrides=overrides,
        environment=environment,
        executed=done,
        skipped=skipped,
        failed=failed,
        results_in_output_dir=total,
    )
    serialize.write_json(os.path.join(args.output_dir, "run_manifest.json"), manifest)

    log(f"done: {len(done)} executed, {len(skipped)} skipped, {len(failed)} failed")
    if failed:
        log("failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
