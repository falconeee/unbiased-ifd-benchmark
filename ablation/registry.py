"""The ablation grid requested by Reviewer #2, and the JSON document it produces.

Two arms sharing one cell
-------------------------
**R2.2 -- depth.** ``depth in (1, 2, 3, 5)`` with every kernel fixed at 3.

**R2.3 -- receptive field.** ``first_kernel in (3, 7, 11, 64)`` on the fixed ``depth=3``
backbone; 11 matches AlexNet's first kernel, 64 matches LeNet's.

``depth=3, first_kernel=3`` belongs to both arms, so the grid is 7 distinct variants rather
than 8. It is trained and stored once (``study="both"``) and appears in both tables of the
report.

Protocol
--------
Identical to the published 1D-CNN row of Table 4 on each dataset: Adam, lr 3e-4, batch 64,
100 epochs, multiround unbiased folds (8 rounds x 4 folds on all three datasets). The
training loop is :class:`src.experiment.DeepLearningExperiment` unchanged, which means a 20%
validation split used only for logging ``val_loss`` and **no early stopping** -- the tested
model is the last epoch's, exactly as in the benchmark. Note that Section 2.2 of the
manuscript currently describes a 10% split with early stopping, which is not what the
benchmark code does; the ablation follows the code so its numbers stay comparable with the
already-published 1D-CNN results.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence, Tuple

#: Datasets the coauthor selected: PU (where the reversal was observed) plus both CWRU
#: variants (where it was not), so the ablation can show whether any trend is dataset
#: specific. All three use 8 rounds x 4 folds under ``src/fold_designs.json``.
DATASET_NAMES: Tuple[str, ...] = ("PU", "CWRU12k", "CWRU48k")

STUDIES: Tuple[str, ...] = ("depth", "kernel")

SUITE = "cnn1d_ablation"
MODEL_CLASS = "AblationCNN1D"

#: Hyperparameters of the ``cnn1d`` experiment on these datasets in ``src/experiments.json``
#: (verified identical across PU, CWRU12k and CWRU48k).
BASELINE_BATCH_SIZE = 64
BASELINE_LR = 3e-4
BASELINE_NUM_EPOCHS = 100

#: The published ``cnn1d`` experiment each dataset's variants should be compared against.
BASELINE_EXPERIMENT = {
    "PU": "multi_round/PU/cnn1d_pu",
    "CWRU12k": "multi_round/CWRU12k/cnn1d_12k",
    "CWRU48k": "multi_round/CWRU48k/cnn1d_vibration",
}

DEPTH_ARM: Tuple[int, ...] = (1, 2, 3, 5)
KERNEL_ARM: Tuple[int, ...] = (3, 7, 11, 64)
SHARED_DEPTH = 3
SHARED_KERNEL = 3

KERNEL_NOTE = {
    3: "baseline kernel",
    7: "first kernel of the published 1D-CNN and of the ResNet-18 stem",
    11: "first kernel of AlexNet",
    64: "first kernel of LeNet and of the BiLSTM stem",
}


@dataclass(frozen=True)
class AblationSpec:
    """One variant on one dataset.

    Deliberately duck-type compatible with :class:`src.registry.ExperimentSpec` for the
    fields the shared serialisation touches (``experiment_name``, ``description``, ``slug``,
    ``suite``, ``dataset``, ``protocol``, ``training_stage``, ``is_autoencoder``).
    """

    key: str
    dataset: str
    study: str
    depth: int
    first_kernel: int
    kernel: int
    batch_size: int
    lr: float
    num_epochs: int

    suite: str = SUITE
    protocol: str = "multi_round"
    model_class: str = MODEL_CLASS
    pretrain_epochs: int = 0
    is_autoencoder: bool = False
    training_stage: str = "classifier"

    @property
    def experiment_name(self) -> str:
        return f"cnn1d_ablation_{self.key}_{self.dataset.lower()}"

    @property
    def slug(self) -> str:
        cleaned = re.sub(r"[^\w\-.]+", "_", self.experiment_name.strip())
        return re.sub(r"_+", "_", cleaned).strip("_")

    @property
    def review_comment(self) -> str:
        return {"depth": "R2.2", "kernel": "R2.3", "both": "R2.2+R2.3"}[self.study]

    @property
    def in_depth_arm(self) -> bool:
        return self.study in ("depth", "both")

    @property
    def in_kernel_arm(self) -> bool:
        return self.study in ("kernel", "both")

    @property
    def description(self) -> str:
        blocks = "block" if self.depth == 1 else "blocks"
        note = KERNEL_NOTE.get(self.first_kernel)
        suffix = f"; {note}" if note and self.first_kernel != 3 else ""
        return (
            f"{self.review_comment} ablation on {self.dataset}: {self.depth} conv {blocks}, "
            f"first kernel {self.first_kernel}, remaining kernels {self.kernel}{suffix}"
        )

    def as_dict(self) -> OrderedDict:
        return OrderedDict(
            variant=self.key,
            study=self.study,
            depth=self.depth,
            first_kernel=self.first_kernel,
            kernel=self.kernel,
        )


def _variant_key(depth: int, first_kernel: int) -> str:
    return f"d{depth}_k{first_kernel}"


def _grid() -> List[Tuple[str, str, int, int]]:
    """``(key, study, depth, first_kernel)`` for the 7 distinct variants, in report order."""
    rows: List[Tuple[str, str, int, int]] = []
    for depth in DEPTH_ARM:
        study = "both" if depth == SHARED_DEPTH else "depth"
        rows.append((_variant_key(depth, SHARED_KERNEL), study, depth, SHARED_KERNEL))
    for first_kernel in KERNEL_ARM:
        if first_kernel == SHARED_KERNEL:
            continue  # already emitted by the depth arm as the shared cell
        rows.append(
            (_variant_key(SHARED_DEPTH, first_kernel), "kernel", SHARED_DEPTH, first_kernel)
        )
    return rows


VARIANTS: Tuple[Tuple[str, str, int, int], ...] = tuple(_grid())
VARIANT_KEYS: Tuple[str, ...] = tuple(row[0] for row in VARIANTS)


def build_registry() -> List[AblationSpec]:
    specs: List[AblationSpec] = []
    for dataset in DATASET_NAMES:
        for key, study, depth, first_kernel in VARIANTS:
            specs.append(
                AblationSpec(
                    key=key,
                    dataset=dataset,
                    study=study,
                    depth=depth,
                    first_kernel=first_kernel,
                    kernel=SHARED_KERNEL,
                    batch_size=BASELINE_BATCH_SIZE,
                    lr=BASELINE_LR,
                    num_epochs=BASELINE_NUM_EPOCHS,
                )
            )
    return specs


REGISTRY: List[AblationSpec] = build_registry()


def select(
    datasets: Sequence[str] = DATASET_NAMES,
    keys: Sequence[str] = VARIANT_KEYS,
    studies: Optional[Sequence[str]] = None,
) -> Iterator[AblationSpec]:
    """Yield the selected variants in registry order.

    Selecting by ``study`` includes the shared ``d3_k3`` cell in either arm.
    """
    for spec in REGISTRY:
        if spec.dataset not in datasets or spec.key not in keys:
            continue
        if studies is not None and not any(
            (study == "depth" and spec.in_depth_arm)
            or (study == "kernel" and spec.in_kernel_arm)
            for study in studies
        ):
            continue
        yield spec


# ------------------------------------------------------------------------- document
PROTOCOL_NOTE = (
    "Trained with src.experiment.DeepLearningExperiment unchanged, i.e. the benchmark's own "
    "protocol: 20% of each training fold held out for val_loss logging only, no early "
    "stopping and no checkpoint selection (the tested model is the last epoch's). This "
    "deviates from the 10%-split-with-early-stopping described in Section 2.2 of the "
    "manuscript, and was chosen so these numbers remain directly comparable with the "
    "already-published 1D-CNN row of Table 4."
)

STRUCTURE_NOTE = (
    "Blocks 0..depth-2 pool with MaxPool1d(2); the last block pools with "
    "AdaptiveMaxPool1d(16). At depth 3 this is exactly src.models.CNN1D with its 7-5-3 "
    "kernel schedule replaced by the variant's. Parameter count is not held constant across "
    "depths -- see the architecture block."
)


def ablation_document(
    *,
    spec: AblationSpec,
    run_info: OrderedDict,
    architecture: OrderedDict,
    configuration: OrderedDict,
    results: OrderedDict,
    fold_design: Optional[dict] = None,
) -> OrderedDict:
    """Same schema as ``src.serialize.experiment_document`` plus the ablation fields.

    ``source`` points at the review comment instead of a v0 notebook, since these
    experiments have no notebook ancestor.
    """
    doc = OrderedDict()
    doc["experiment_name"] = spec.experiment_name
    doc["dataset"] = spec.dataset
    doc["suite"] = spec.suite
    doc["protocol"] = spec.protocol
    doc["model"] = spec.model_class
    doc["study"] = spec.study
    doc["variant"] = spec.key
    doc["description"] = spec.description
    doc["status"] = "executed"
    doc["source"] = OrderedDict(
        review_comment=spec.review_comment,
        review_file="review/reviwers_comments.md",
        baseline_experiment=BASELINE_EXPERIMENT.get(spec.dataset),
        produced_by="run_experiments_ablation.py",
    )
    doc["run"] = run_info
    doc["architecture"] = architecture
    doc["configuration"] = configuration
    doc["results"] = results
    if fold_design:
        doc["fold_design"] = fold_design
    doc["notes"] = OrderedDict(protocol=PROTOCOL_NOTE, structure=STRUCTURE_NOTE)
    return doc
