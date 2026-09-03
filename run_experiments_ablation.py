#!/usr/bin/env python3
"""Run the controlled CNN1D ablations requested by Reviewer #2 and save one JSON per variant.

Reviewer #2 objects that the AlexNet vs. ResNet-18 comparison cannot isolate depth, because
the two architectures differ along several axes at once, and that the receptive-field
explanation is untested. This script answers both with a single parametric family built on
the paper's own 1D-CNN (``ablation/models.py``):

    R2.2  depth arm    depth in (1, 2, 3, 5), every kernel fixed at 3
    R2.3  kernel arm   first kernel in (3, 7, 11, 64), depth fixed at 3

``d3_k3`` belongs to both arms, so the grid is 7 variants x 3 datasets (PU, CWRU12k,
CWRU48k) = 21 experiments, each 8 rounds x 4 folds = 672 fold trainings.

Everything else matches the published 1D-CNN row of Table 4 -- same unbiased multiround
folds, Adam, lr 3e-4, batch 64, 100 epochs, and the benchmark's own training loop, which
means a 20% validation split used only for logging and **no early stopping**. Results use
the schema of ``run_experiments.py`` plus an ``architecture`` block recording parameter
count and theoretical receptive field per variant.

Setup (same environment as the main benchmark)
----------------------------------------------
    uv venv --no-project
    uv pip install -r requirements.tx

Examples
--------
    # everything, resumable, in the background
    nohup uv run --no-project python run_experiments_ablation.py \
        --all --resume --keep-going > run_ablation.out 2>&1 &

    # the dataset the reviewer asked about
    uv run --no-project python run_experiments_ablation.py --dataset PU

    # one arm only (includes the shared d3_k3 cell)
    uv run --no-project python run_experiments_ablation.py --dataset PU --study depth

    # quick smoke test
    uv run --no-project python run_experiments_ablation.py --dataset CWRU12k \
        --variant d1_k3 --epochs 2 --max-rounds 1

    # what would run
    uv run --no-project python run_experiments_ablation.py --all --list

    # rebuild the tables and statistics from results already on disk
    uv run --no-project python run_experiments_ablation.py --report
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import traceback
import warnings
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Dict, List, Optional

warnings.filterwarnings("ignore")

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

# Logging, the tee and the environment probe are identical to the main runner; importing
# them keeps the two scripts from drifting apart.
from run_experiments import Tee, environment_info, log, resolve_device  # noqa: E402


# ------------------------------------------------------------------------------ CLI
def build_parser() -> argparse.ArgumentParser:
    from ablation.registry import DATASET_NAMES, STUDIES, VARIANT_KEYS

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sel = parser.add_argument_group("selection")
    sel.add_argument("--all", action="store_true",
                     help="run every dataset and variant (same as passing no filter)")
    sel.add_argument("--dataset", action="append", choices=DATASET_NAMES,
                     help="restrict to a dataset (repeatable)")
    sel.add_argument("--variant", action="append", choices=VARIANT_KEYS,
                     help="restrict to a variant, e.g. d3_k11 (repeatable)")
    sel.add_argument("--study", action="append", choices=STUDIES,
                     help="restrict to an ablation arm; either arm includes the shared "
                          "d3_k3 cell (repeatable)")
    sel.add_argument("--list", action="store_true", help="print the selection and exit")

    paths = parser.add_argument_group("paths")
    paths.add_argument("--data-root", default=os.path.join(REPO_ROOT, "data"),
                       help="where raw/converted datasets and fold caches live -- share this "
                            "with run_experiments.py to avoid re-downloading "
                            "(default: %(default)s)")
    paths.add_argument("--output-dir", default=os.path.join(REPO_ROOT, "results_ablation"),
                       help="where the result JSONs and the report are written "
                            "(default: %(default)s)")
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
                     help="skip variants whose result JSON already exists")
    run.add_argument("--keep-going", action="store_true",
                     help="carry on to the next variant when one raises")
    run.add_argument("--no-artifacts", action="store_true",
                     help="do not write loss curves and model checkpoints")
    run.add_argument("--folds-source", choices=("notebook", "generate"), default="notebook",
                     help="multiround folds: rebuild from the notebooks' printed design "
                          "(default) or run the full combination search, which needs ~15 GB "
                          "of RAM for CWRU12k/PU")

    override = parser.add_argument_group("overrides (for smoke tests)")
    override.add_argument("--epochs", type=int, default=None, help="override num_epochs")
    override.add_argument("--batch-size", type=int, default=None, help="override batch_size")
    override.add_argument("--max-rounds", type=int, default=None,
                          help="run at most this many of the 8 rounds")

    report = parser.add_argument_group("reporting")
    report.add_argument("--report", action="store_true",
                        help="rebuild report.md/report.tex/report.json from the results "
                             "already in --output-dir and exit without training")
    return parser


# ------------------------------------------------------------------------ execution
#: The benchmark's own value (``DeepLearningExperiment.val_split`` default), used only to
#: log ``val_loss``. Not 0.10 as Section 2.2 of the manuscript states -- see the report.
VAL_SPLIT = 0.2


def experiment_configuration(spec, input_length: int, num_classes: int, args) -> OrderedDict:
    """The training configuration, recorded verbatim in the result JSON."""
    cfg = OrderedDict()
    cfg["batch_size"] = args.batch_size if args.batch_size is not None else spec.batch_size
    cfg["lr"] = spec.lr
    cfg["num_epochs"] = args.epochs if args.epochs is not None else spec.num_epochs
    cfg["pretrain_epochs"] = spec.pretrain_epochs
    cfg["optimizer"] = "Adam"
    cfg["loss"] = "CrossEntropyLoss"
    # Pinned explicitly rather than left to the DeepLearningExperiment default, so the
    # protocol is legible from the JSON alone (see ablation.registry.PROTOCOL_NOTE).
    cfg["val_split"] = VAL_SPLIT
    cfg["early_stopping"] = False
    cfg["checkpoint_selection"] = False
    cfg["input_length"] = input_length
    cfg["num_classes"] = num_classes
    return cfg


def build_variant(spec, input_length: int, num_classes: int):
    from ablation.models import AblationCNN1D

    return AblationCNN1D(
        input_length=input_length,
        num_classes=num_classes,
        depth=spec.depth,
        first_kernel=spec.first_kernel,
        kernel=spec.kernel,
    )


def run_one_round(spec, cfg, fold_idxs, X, y, input_length, num_classes, args,
                  artifacts_dir: str, round_label: str):
    """Build the variant and run one full cross validation; returns (results, experiment)."""
    from src.experiment import DeepLearningExperiment, seed_everything

    # Same contract as run_experiments.run_one_round: re-seed before building so every
    # round of every variant starts from a deterministic initialisation.
    seed_everything(args.seed)
    model = build_variant(spec, input_length, num_classes)

    experiment = DeepLearningExperiment(
        name=f"{spec.experiment_name}{round_label}",
        description=spec.description,
        data_fold_idxs=fold_idxs,
        model=model,
        batch_size=cfg["batch_size"],
        lr=cfg["lr"],
        num_epochs=cfg["num_epochs"],
        pretrain_epochs=cfg["pretrain_epochs"],
        val_split=cfg["val_split"],
        output_dir=artifacts_dir,
        device=args.device,
        X=X,
        y=y,
        seed=args.seed,
        save_artifacts=not args.no_artifacts,
    )
    return experiment.run(), experiment


def run_experiment(spec, context, args) -> OrderedDict:
    """Run every round of one variant and return the document to serialise."""
    from src import serialize
    from ablation.models import architecture_block
    from ablation.registry import ablation_document

    X, y = context["X"], context["y"]
    input_length, num_classes = context["input_length"], context["num_classes"]
    rounds_folds = context["rounds"]
    cfg = experiment_configuration(spec, input_length, num_classes, args)

    architecture = architecture_block(
        build_variant(spec, input_length, num_classes), input_length
    )
    log(f"    depth={architecture['depth']} kernels={architecture['kernels_per_block']} "
        f"params={architecture['num_parameters']:,} "
        f"receptive_field={architecture['receptive_field_samples']} samples")

    artifacts_dir = os.path.join(args.artifacts_dir, spec.dataset, spec.slug)
    started = time.time()

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

    return ablation_document(
        spec=spec,
        run_info=run_info,
        architecture=architecture,
        configuration=cfg,
        results=results_block,
        fold_design=context.get("fold_design"),
    )


def dataset_context(dataset: str, args) -> Dict:
    """Load the dataset and its multiround folds once, shared by all of its variants."""
    import numpy as np

    from src import folds as folds_mod
    from src.datasets import DATASETS, load_deep_dataset
    from src.experiment import load_xy

    dataset_spec = DATASETS[dataset]

    log(f"  loading dataset {dataset} (multi_round)")
    deep_dataset = load_deep_dataset(dataset_spec, args.data_root, download=not args.no_download)
    X, y, _ = load_xy(deep_dataset)
    input_length = int(X.shape[-1])
    num_classes = int(len(np.unique(y)))
    log(f"  {len(X)} samples, input_length={input_length}, num_classes={num_classes}")

    rounds, folds_source = folds_mod.build_folds(
        dataset_spec, deep_dataset, "multi_round", args.data_root, args.folds_source
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
    designs = folds_mod.load_fold_designs()
    if dataset in designs:
        context["fold_design"] = designs[dataset]
    return context


# ----------------------------------------------------------------------- reporting
def rebuild_indices(output_dir: str) -> int:
    """Regenerate ``_index.json`` per dataset and ``index.json`` at the root."""
    from src import serialize

    all_rows = []
    for ds_dir in sorted(glob.glob(os.path.join(output_dir, "*"))):
        if not os.path.isdir(ds_dir) or os.path.basename(ds_dir).startswith("_"):
            continue
        dataset = os.path.basename(ds_dir)
        rows = []
        for path in sorted(glob.glob(os.path.join(ds_dir, "*.json"))):
            if os.path.basename(path) == "_index.json":
                continue
            with open(path, encoding="utf-8") as fh:
                doc = json.load(fh)
            summary = doc.get("results", {}).get("summary", {})
            architecture = doc.get("architecture", {})
            rows.append(OrderedDict(
                file=os.path.basename(path),
                experiment_name=doc.get("experiment_name"),
                variant=doc.get("variant"),
                study=doc.get("study"),
                depth=architecture.get("depth"),
                first_kernel=architecture.get("first_kernel"),
                num_parameters=architecture.get("num_parameters"),
                receptive_field_samples=architecture.get("receptive_field_samples"),
                rounds_executed=summary.get("rounds_executed"),
                mean_accuracy=summary.get("mean_accuracy"),
                std_accuracy=summary.get("std_accuracy"),
                mean_f1_score=summary.get("mean_f1_score"),
                std_f1_score=summary.get("std_f1_score"),
            ))
            all_rows.append(OrderedDict(dataset=dataset, **rows[-1]))
        if rows:
            serialize.write_json(
                os.path.join(ds_dir, "_index.json"),
                OrderedDict(dataset=dataset, suite="cnn1d_ablation",
                            source="review/reviwers_comments.md (R2.2, R2.3)",
                            num_experiments=len(rows), experiments=rows),
            )
    if all_rows:
        serialize.write_json(
            os.path.join(output_dir, "index.json"),
            OrderedDict(total_experiments=len(all_rows), experiments=all_rows),
        )
    return len(all_rows)


def write_report(output_dir: str) -> int:
    """Build report.json / report.md / report.tex. Returns the number of datasets covered."""
    from ablation import report as ablation_report
    from src import serialize

    report, markdown, latex = ablation_report.build(output_dir)
    if not report:
        log("no ablation results found; report not written")
        return 0

    serialize.write_json(os.path.join(output_dir, "report.json"), report)
    for name, text in (("report.md", markdown), ("report.tex", latex)):
        with open(os.path.join(output_dir, name), "w", encoding="utf-8") as fh:
            fh.write(text)
    log(f"report written for {len(report)} dataset(s): "
        f"{os.path.relpath(os.path.join(output_dir, 'report.md'), REPO_ROOT)}")
    return len(report)


# ---------------------------------------------------------------------------- main
def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    from ablation.registry import DATASET_NAMES, VARIANT_KEYS, select

    if args.report:
        return 0 if write_report(args.output_dir) else 1

    datasets = tuple(args.dataset) if args.dataset else DATASET_NAMES
    variants = tuple(args.variant) if args.variant else VARIANT_KEYS
    studies = tuple(args.study) if args.study else None
    selection = list(select(datasets, variants, studies))

    if args.list:
        print(f"{len(selection)} experiment(s) selected\n")
        print(f"{'dataset':<9}{'variant':<9}{'study':<7}{'depth':<7}{'k1':<5}experiment")
        for spec in selection:
            print(f"{spec.dataset:<9}{spec.key:<9}{spec.study:<7}{spec.depth:<7}"
                  f"{spec.first_kernel:<5}{spec.experiment_name}")
        return 0

    if not selection:
        parser.error("no experiment matches the given filters")

    os.makedirs(args.output_dir, exist_ok=True)
    if args.artifacts_dir is None:
        args.artifacts_dir = os.path.join(args.output_dir, "_artifacts")
    if args.log_file is None:
        args.log_file = os.path.join(args.output_dir, "run.log")

    sys.stdout = Tee(sys.stdout, args.log_file)
    sys.stderr = Tee(sys.stderr, args.log_file)

    from src import serialize

    device = resolve_device(args.device)
    args.device = device
    run_started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    environment = environment_info(device)
    overrides = OrderedDict(
        (name, getattr(args, name))
        for name in ("epochs", "batch_size", "max_rounds")
        if getattr(args, name) is not None
    )

    log(f"{len(selection)} ablation experiment(s) selected | device={device} | seed={args.seed}")
    log(f"output-dir={args.output_dir}  data-root={args.data_root}")
    if overrides:
        log(f"overrides: {dict(overrides)}")

    done, skipped, failed = [], [], []
    by_dataset: "OrderedDict[str, list]" = OrderedDict()
    for spec in selection:
        by_dataset.setdefault(spec.dataset, []).append(spec)

    for dataset, specs in by_dataset.items():
        target_dir = os.path.join(args.output_dir, dataset)

        pending = specs
        if args.resume:
            pending = [s for s in specs
                       if not os.path.exists(os.path.join(target_dir, s.slug + ".json"))]
            for spec in specs:
                if spec not in pending:
                    skipped.append(f"{dataset}/{spec.key}")
        if not pending:
            log(f"{dataset}: nothing to do (all results present)")
            continue

        log(f"=== {dataset} -- {len(pending)} variant(s)")
        try:
            context = dataset_context(dataset, args)
        except Exception as exc:  # noqa: BLE001
            log(f"!! {dataset}: failed to prepare dataset/folds: {exc}")
            traceback.print_exc()
            failed.extend(f"{dataset}/{s.key}" for s in pending)
            if args.keep_going:
                continue
            return 1
        context["run_started_at"] = run_started_at
        context["device"] = device
        context["environment"] = environment
        context["overrides"] = overrides

        os.makedirs(target_dir, exist_ok=True)
        for spec in pending:
            label = f"{dataset}/{spec.key} ({spec.experiment_name})"
            log(f"--- {label}")
            try:
                document = run_experiment(spec, context, args)
            except Exception as exc:  # noqa: BLE001
                log(f"!! {label} failed: {exc}")
                traceback.print_exc()
                failed.append(f"{dataset}/{spec.key}")
                if args.keep_going:
                    continue
                return 1
            path = os.path.join(target_dir, spec.slug + ".json")
            serialize.write_json(path, document)
            summary = document["results"].get("summary", {})
            log(f"    saved {os.path.relpath(path, REPO_ROOT)} | "
                f"acc={summary.get('mean_accuracy')} f1={summary.get('mean_f1_score')}")
            done.append(f"{dataset}/{spec.key}")

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

    write_report(args.output_dir)

    log(f"done: {len(done)} executed, {len(skipped)} skipped, {len(failed)} failed")
    if failed:
        log("failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
