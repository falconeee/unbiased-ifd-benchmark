"""Fold generation for the unbiased protocol.

Single round folds come straight from the dataset's ``GroupDataset`` (the same thing
``FoldIdxGeneratorUnbiased.generate_folds()`` returns).

Multi round folds can be obtained two ways:

``notebook`` (default)
    Rebuild them from the round x fold design the v0 notebooks printed, stored in
    ``src/fold_designs.json``. Instant, and guaranteed to be
    the same folds the published results were produced with.

``generate``
    Run ``FoldIdxGeneratorUnbiased.compute_combinations`` for real. Correct, but for
    CWRU12k/PU it materialises ``list(combinations(256, 4))`` = 174,792,640 tuples
    (~15 GB of RAM) before shuffling, so it only runs on a very large machine.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import numpy as np

from signalAI.utils.fold_idx_generator import FoldIdxGeneratorUnbiased

from .datasets import DatasetSpec

FOLD_DESIGNS_PATH = os.path.join(os.path.dirname(__file__), "fold_designs.json")


def _groups_dir(data_root: str) -> str:
    return os.path.join(data_root, "grouping")


def _group_array(group_class, deep_dataset, dataset_name: str, data_root: str) -> np.ndarray:
    """Same call ``FoldIdxGeneratorUnbiased`` makes, with a configurable cache directory."""
    grouper = group_class(
        deep_dataset,
        custom_name="CustomGroup" + dataset_name,
        groups_dir=_groups_dir(data_root),
    )
    return grouper.groups()


def single_round_folds(spec: DatasetSpec, deep_dataset, data_root: str) -> np.ndarray:
    """Fold index per sample, as ``FoldIdxGeneratorUnbiased(...).generate_folds()`` returns."""
    folds = np.asarray(_group_array(spec.group_class, deep_dataset, spec.group_name, data_root))
    _validate_single_round(folds)
    return folds


def _validate_single_round(folds: np.ndarray) -> None:
    """The validation ``generate_folds`` performs (it is correct for the 1-D case)."""
    found = set(int(f) for f in np.unique(folds))
    expected = set(range(max(found) + 1))
    missing = expected - found
    if missing:
        raise ValueError(f"Missing folds: {missing} (expected {expected}, got {found})")
    if 0 not in found:
        raise ValueError("Fold 0 must be present but was not found.")


# ------------------------------------------------------------------- multi round folds
def _multiround_group_array(spec: DatasetSpec, deep_dataset, data_root: str) -> np.ndarray:
    # ``generate_folds_unbiased_multiround`` appends "_multiround" before grouping, so the
    # cache file name has to match to reuse it.
    name = spec.multiround_group_name + "_multiround"
    return np.asarray(_group_array(spec.multiround_group_class, deep_dataset, name, data_root))


def _labels(deep_dataset) -> np.ndarray:
    return np.array([sample["metainfo"]["label"] for sample in deep_dataset])


def load_fold_designs() -> Dict[str, dict]:
    with open(FOLD_DESIGNS_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def multiround_folds_from_design(spec: DatasetSpec, deep_dataset, data_root: str) -> List[np.ndarray]:
    """Rebuild the notebooks' multiround folds from the printed round x fold design.

    ``FoldIdxGeneratorUnbiased.print_combinations`` renders each raw group ``"<label>
    <condition>"`` as ``class_def[label] + " " + condition_def[condition]``. Both maps are
    injective for every dataset here, so the rendering can be inverted back to the raw
    group strings and turned into the same ``fold_map`` the generator builds.
    """
    designs = load_fold_designs()
    if spec.name not in designs:
        raise KeyError(f"no recorded multiround fold design for {spec.name}")
    design = designs[spec.name]

    label_of = _invert(spec.class_def, f"{spec.name}.class_def")
    condition_of = _invert(spec.condition_def, f"{spec.name}.condition_def")

    groups = _multiround_group_array(spec, deep_dataset, data_root)
    available = set(np.unique(groups).tolist())

    folds_multiround: List[np.ndarray] = []
    for round_design in design["rounds"]:
        fold_map: Dict[str, int] = {}
        for fold in round_design["folds"]:
            for rendered in fold["groups"]:
                raw = _raw_group(rendered, label_of, condition_of)
                if raw not in available:
                    raise ValueError(
                        f"{spec.name}: group {raw!r} (printed as {rendered!r}) is not present in the "
                        f"dataset; the recorded fold design does not match this build of the dataset"
                    )
                fold_map[raw] = fold["fold"]
        missing = available - set(fold_map)
        if missing:
            raise ValueError(
                f"{spec.name} round {round_design['round']}: groups {sorted(missing)} are not "
                f"assigned to any fold in the recorded design"
            )
        folds_multiround.append(np.array([fold_map[gp] for gp in groups]))
    return folds_multiround


def _invert(mapping: Dict, what: str) -> Dict[str, str]:
    inverted: Dict[str, str] = {}
    for key, value in mapping.items():
        if value in inverted:
            raise ValueError(f"{what} is not injective: {value!r} maps from more than one key")
        inverted[value] = str(key)
    return inverted


def _raw_group(rendered: str, label_of: Dict[str, str], condition_of: Dict[str, str]) -> str:
    label_token, condition_token = rendered.split(" ", 1)
    return f"{label_of[label_token]} {condition_of[condition_token]}"


def multiround_folds_generated(spec: DatasetSpec, deep_dataset, data_root: str) -> List[np.ndarray]:
    """Run the real combination search from ``signalAI`` (memory hungry -- see module docs)."""
    generator = FoldIdxGeneratorUnbiased(
        deep_dataset,
        spec.multiround_group_class,
        dataset_name=spec.multiround_group_name,
        multiround=True,
        class_def=spec.class_def,
        condition_def=spec.condition_def,
    )
    y = _labels(deep_dataset)
    groups = _multiround_group_array(spec, deep_dataset, data_root)

    n_splits = int(np.unique(groups).shape[0] / np.unique(y).shape[0])
    n_repeats = int(np.ceil(30 / n_splits))
    print("Per round splits: ", n_splits)
    print("Number of repeats: ", n_repeats)
    # NOTE: ``generate_folds()`` cannot be used here -- with ``multiround=True`` it ends up
    # calling ``max()`` on a list of numpy arrays, which raises
    # "The truth value of an array with more than one element is ambiguous".
    return generator.compute_combinations(y, groups, n_splits, n_repeats, generator.random_state)


def build_folds(
    spec: DatasetSpec,
    deep_dataset,
    protocol: str,
    data_root: str,
    folds_source: str = "notebook",
) -> Tuple[List[np.ndarray], str]:
    """Return ``(rounds, source)`` where ``rounds`` is a list of fold-index arrays.

    A single round run yields a one element list, so callers can treat both the same way.
    """
    if protocol == "single_round":
        return [single_round_folds(spec, deep_dataset, data_root)], "group_dataset"
    if not spec.has_multiround:
        raise ValueError(f"{spec.name} has no multiround grouping defined")
    if folds_source == "generate":
        return multiround_folds_generated(spec, deep_dataset, data_root), "generated"
    return multiround_folds_from_design(spec, deep_dataset, data_root), "notebook_design"
