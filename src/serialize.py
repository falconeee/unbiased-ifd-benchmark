"""Turn ``ExperimentResults`` into the JSON schema used under ``results/``.

The layout mirrors the archived ``v0_experiments/v0_results/``, so a freshly run experiment
and the one extracted from the v0 notebooks can be diffed field by field. What the notebooks
could not provide -- every epoch instead of every fifth, per-fold confusion matrices,
precision/recall/roc_auc per fold, wall times -- is added on top.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from typing import Dict, List, Optional

import numpy as np

from signalAI.utils.experiment_result import ExperimentResults


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def write_json(path, payload) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, cls=NumpyEncoder)
        fh.write("\n")


def _confusion_stats(matrix) -> OrderedDict:
    cm = np.asarray(matrix)
    total = int(cm.sum())
    stats = OrderedDict()
    stats["overall_accuracy"] = float(np.trace(cm) / total) if total else None
    stats["total_samples"] = total
    stats["matrix"] = cm.tolist()
    return stats


def per_fold_metrics(results: ExperimentResults) -> List[OrderedDict]:
    rows = []
    for i, fold in enumerate(results.folds):
        row = OrderedDict()
        row["fold"] = i + 1
        row["accuracy"] = fold.metrics.get("accuracy")
        row["f1_score"] = fold.metrics.get("f1")
        row["precision"] = fold.metrics.get("precision")
        row["recall"] = fold.metrics.get("recall")
        if fold.metrics.get("roc_auc") is not None:
            row["roc_auc"] = fold.metrics["roc_auc"]
        rows.append(row)
    return rows


def folds_training_log(
    results: ExperimentResults,
    history: Dict[int, Dict[str, list]],
    training_stage: str,
    epochs_logged_for: int,
) -> List[OrderedDict]:
    rows = []
    total = len(results.folds)
    for i, fold in enumerate(results.folds):
        hist = history.get(fold.fold_index, {})
        row = OrderedDict()
        row["fold"] = i + 1
        row["total_folds"] = total
        row["fold_index"] = fold.fold_index
        row["training_stage"] = training_stage
        row["epochs_logged_for"] = epochs_logged_for
        row["accuracy"] = fold.metrics.get("accuracy")
        row["f1_score"] = fold.metrics.get("f1")
        row["confusion_matrix"] = np.asarray(fold.confusion_matrix).tolist()
        entry = OrderedDict()
        if hist.get("pretrain"):
            entry["pretrain"] = hist["pretrain"]
        if hist.get("supervised"):
            entry["supervised"] = hist["supervised"]
        row["training_history"] = entry
        rows.append(row)
    return rows


def summary_block(results: ExperimentResults) -> OrderedDict:
    om = results.overall_metrics
    block = OrderedDict()
    block["mean_accuracy"] = om.get("accuracy")
    block["std_accuracy"] = om.get("std_accuracy")
    block["mean_f1_score"] = om.get("mean_f1")
    block["std_f1_score"] = om.get("std_f1")
    block["num_folds"] = len(results.folds)
    return block


def single_round_results(
    results: ExperimentResults,
    history: Dict[int, Dict[str, list]],
    training_stage: str,
    epochs_logged_for: int,
    fold_errors: List[dict],
    duration_seconds: float,
) -> OrderedDict:
    block = OrderedDict()
    block["summary"] = summary_block(results)
    block["per_fold_metrics"] = per_fold_metrics(results)
    block["confusion_matrix"] = _confusion_stats(results.overall_metrics["confusion_matrix"])
    block["folds_training_log"] = folds_training_log(
        results, history, training_stage, epochs_logged_for
    )
    block["duration_seconds"] = round(duration_seconds, 3)
    if fold_errors:
        block["fold_errors"] = fold_errors
    return block


def multi_round_results(rounds: List[OrderedDict], total_rounds_planned: int,
                        duration_seconds: float) -> OrderedDict:
    accuracies = [r["summary"]["mean_accuracy"] for r in rounds if r.get("summary")]
    f1_scores = [r["summary"]["mean_f1_score"] for r in rounds if r.get("summary")]

    block = OrderedDict()
    block["total_rounds_planned"] = total_rounds_planned
    summary = OrderedDict()
    summary["rounds_executed"] = len(rounds)
    summary["mean_accuracy"] = float(np.mean(accuracies)) if accuracies else None
    summary["std_accuracy"] = float(np.std(accuracies)) if accuracies else None
    summary["mean_f1_score"] = float(np.mean(f1_scores)) if f1_scores else None
    summary["std_f1_score"] = float(np.std(f1_scores)) if f1_scores else None
    block["summary"] = summary
    block["rounds"] = rounds
    block["duration_seconds"] = round(duration_seconds, 3)
    return block


def round_block(
    round_number: int,
    total_rounds: int,
    results: ExperimentResults,
    history: Dict[int, Dict[str, list]],
    training_stage: str,
    epochs_logged_for: int,
    fold_errors: List[dict],
    duration_seconds: float,
) -> OrderedDict:
    block = OrderedDict()
    block["round"] = round_number
    block["total_rounds"] = total_rounds
    block["summary"] = summary_block(results)
    block["mean_accuracy"] = results.overall_metrics.get("accuracy")
    block["per_fold_metrics"] = per_fold_metrics(results)
    block["confusion_matrix"] = _confusion_stats(results.overall_metrics["confusion_matrix"])
    block["folds"] = folds_training_log(results, history, training_stage, epochs_logged_for)
    block["duration_seconds"] = round(duration_seconds, 3)
    if fold_errors:
        block["fold_errors"] = fold_errors
    return block


def experiment_document(
    *,
    spec: dict,
    dataset: str,
    suite: str,
    protocol: str,
    model_class: str,
    run_info: OrderedDict,
    configuration: OrderedDict,
    results: OrderedDict,
    fold_design: Optional[dict] = None,
    notes: Optional[str] = None,
) -> OrderedDict:
    doc = OrderedDict()
    doc["experiment_name"] = spec["experiment_name"]
    if spec.get("notebook_experiment_name_template"):
        doc["experiment_name_template"] = spec["notebook_experiment_name_template"]
    doc["dataset"] = dataset
    doc["suite"] = suite
    doc["protocol"] = protocol
    doc["model"] = model_class
    doc["description"] = spec["description"]
    doc["section"] = spec.get("section")
    doc["status"] = "executed"
    doc["source"] = OrderedDict(
        notebook=f"v0_experiments/{suite}_experiments/SignalAI_Framework_{dataset}.ipynb",
        produced_by="run_experiments.py",
    )
    doc["run"] = run_info
    doc["configuration"] = configuration
    doc["results"] = results
    if fold_design:
        doc["fold_design"] = fold_design
    if notes:
        doc["notes"] = notes
    return doc
