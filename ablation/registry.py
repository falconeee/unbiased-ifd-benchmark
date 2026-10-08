"""The ablation grid requested by Reviewer #2, and the JSON document it produces.

Three model families, two arms each
-----------------------------------
Every family (``ablation/models.py``) runs the same two arms on its own architecture:

**R2.2 -- depth.** Depth varies at a fixed first kernel.

**R2.3 -- receptive field.** ``first_kernel in (3, 7, 11, 64)`` at a fixed depth; 7 matches
the 1D-CNN's and ResNet-18's first kernel, 11 AlexNet's, 64 LeNet's.

========  =====================  ========================  ==============================
family    depth arm              kernel arm                shared cell
========  =====================  ========================  ==============================
cnn1d     1, 2, 3, 5 blocks; k3  depth 3; k 3, 7, 11, 64   ``d3_k3``
alexnet   1, 2, 3, 5 convs; k11  depth 5; k 3, 7, 11, 64   ``d5_k11`` = published AlexNet
resnet18  10, 18, 26, 34; k7     depth 18; k 3, 7, 11, 64  ``d18_k7`` = published ResNet18
========  =====================  ========================  ==============================

The shared cell belongs to both arms, so each family has 7 distinct variants rather than 8;
it is trained and stored once (``study="both"``) and appears in both tables of the report.
The 1D-CNN grid follows the response letter (every later kernel 3, shared cell ``d3_k3``);
AlexNet and ResNet-18 are anchored on the published model instead, so their shared cell can
be checked against Table 4.

Protocol
--------
Identical to each family's published row of Table 4 on each dataset: the batch size, learning
rate and epochs come from that row in ``src/experiments.json`` (Adam, lr 3e-4, batch 64,
100 epochs everywhere, except ResNet-18 on PU: batch 128, 25 epochs), on the multiround
unbiased folds (8 rounds x 4 folds on all three datasets). The training loop is
:class:`src.experiment.DeepLearningExperiment` unchanged, which means a 20% validation split
used only for logging ``val_loss`` and **no early stopping** -- the tested model is the last
epoch's, exactly as in the benchmark. Note that Section 2.2 of the manuscript currently
describes a 10% split with early stopping, which is not what the benchmark code does; the
ablation follows the code so its numbers stay comparable with the already-published results.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Iterator, List, Mapping, Optional, Sequence, Tuple

from src import registry as benchmark

#: Datasets the coauthor selected: PU (where the reversal was observed) plus both CWRU
#: variants (where it was not), so the ablation can show whether any trend is dataset
#: specific. All three use 8 rounds x 4 folds under ``src/fold_designs.json``.
DATASET_NAMES: Tuple[str, ...] = ("PU", "CWRU12k", "CWRU48k")

STUDIES: Tuple[str, ...] = ("depth", "kernel")


@dataclass(frozen=True)
class Family:
    """One architecture the two arms are run on."""

    key: str  # also the model key in src/experiments.json, where hyperparameters come from
    label: str  # how the manuscript names it
    model_class: str
    depth_arm: Tuple[int, ...]
    kernel_arm: Tuple[int, ...]
    shared_depth: int
    shared_kernel: int
    depth_unit: str
    kernel_notes: Mapping[int, str] = field(compare=False)
    structure_note: str = field(compare=False)

    @property
    def suite(self) -> str:
        return f"{self.key}_ablation"


CNN1D = Family(
    key="cnn1d",
    label="1D-CNN",
    model_class="AblationCNN1D",
    depth_arm=(1, 2, 3, 5),
    kernel_arm=(3, 7, 11, 64),
    shared_depth=3,
    shared_kernel=3,
    depth_unit="conv blocks",
    kernel_notes={
        3: "baseline kernel",
        7: "first kernel of the published 1D-CNN and of the ResNet-18 stem",
        11: "first kernel of AlexNet",
        64: "first kernel of LeNet and of the BiLSTM stem",
    },
    structure_note=(
        "Blocks 0..depth-2 pool with MaxPool1d(2); the last block pools with "
        "AdaptiveMaxPool1d(16). At depth 3 this is exactly src.models.CNN1D with its 7-5-3 "
        "kernel schedule replaced by the variant's. Parameter count is not held constant "
        "across depths -- see the architecture block."
    ),
)

ALEXNET = Family(
    key="alexnet",
    label="AlexNet",
    model_class="AblationAlexNet1D",
    depth_arm=(1, 2, 3, 5),
    kernel_arm=(3, 7, 11, 64),
    shared_depth=5,
    shared_kernel=11,
    depth_unit="conv layers",
    kernel_notes={
        3: "smaller than the stem stride of 4, so the stem skips one input sample in four",
        7: "first kernel of the published 1D-CNN and of the ResNet-18 stem",
        11: "the published AlexNet stem",
        64: "first kernel of LeNet and of the BiLSTM stem",
    },
    structure_note=(
        "src.models.AlexNet1D truncated to its first `depth` conv layers, with the stem "
        "kernel as a parameter. Widths 64-192-384-256-256 and later kernels 5-3-3-3 are the "
        "published ones; MaxPool1d(3, 2) follows conv 0, conv 1 and the last conv (the "
        "published pattern at depth 5). Stem stride 4 and padding 2, no BatchNorm, "
        "AdaptiveAvgPool1d(6) and the 1024-1024 dropout head are fixed. depth=5, "
        "first_kernel=11 is AlexNet1D weight for weight under the same seed. Parameter count "
        "is not monotonic in depth: the head input is 6 x the last conv's width (384 at "
        "depth 3, 256 at depth 5) -- see the architecture block."
    ),
)

RESNET18 = Family(
    key="resnet18",
    label="ResNet-18",
    model_class="AblationResNet1D",
    depth_arm=(10, 18, 26, 34),
    kernel_arm=(3, 7, 11, 64),
    shared_depth=18,
    shared_kernel=7,
    depth_unit="weighted layers",
    kernel_notes={
        3: "baseline kernel of the 1D-CNN ablation",
        7: "the published ResNet-18 stem",
        11: "first kernel of AlexNet",
        64: "first kernel of LeNet and of the BiLSTM stem",
    },
    structure_note=(
        "src.models.ResNet18 with [n, n, n, n] BasicBlocks per stage and the stem kernel as "
        "parameters; depth is the number of weighted layers, 2 + 8n (ResNet-10/18/26/34). "
        "Stage widths 64-128-256-512, the three stride-2 stages, the stem (stride 2, padding "
        "first_kernel // 2, BatchNorm, MaxPool1d(3, 2)), skip connections, initialisation and "
        "the AdaptiveAvgPool1d(1) + linear head are fixed. depth=18, first_kernel=7 is "
        "ResNet18 weight for weight under the same seed. Parameter count grows with depth -- "
        "see the architecture block."
    ),
)

FAMILIES: "OrderedDict[str, Family]" = OrderedDict(
    (family.key, family) for family in (CNN1D, ALEXNET, RESNET18)
)
FAMILY_KEYS: Tuple[str, ...] = tuple(FAMILIES)


def _benchmark_spec(family: str, dataset: str) -> "benchmark.ExperimentSpec":
    """The published experiment whose hyperparameters a family's variants inherit."""
    for spec in benchmark.REGISTRY[f"multi_round/{dataset}"]:
        if spec.key == family:
            return spec
    raise KeyError(f"no multi_round/{dataset} experiment for model {family!r}")


@dataclass(frozen=True)
class AblationSpec:
    """One variant of one family on one dataset.

    Deliberately duck-type compatible with :class:`src.registry.ExperimentSpec` for the
    fields the shared serialisation touches (``experiment_name``, ``description``, ``slug``,
    ``suite``, ``dataset``, ``protocol``, ``training_stage``, ``is_autoencoder``).
    """

    family: str
    key: str
    dataset: str
    study: str
    depth: int
    first_kernel: int
    kernel: Optional[int]
    batch_size: int
    lr: float
    num_epochs: int

    protocol: str = "multi_round"
    pretrain_epochs: int = 0
    is_autoencoder: bool = False
    training_stage: str = "classifier"

    @property
    def family_spec(self) -> Family:
        return FAMILIES[self.family]

    @property
    def suite(self) -> str:
        return self.family_spec.suite

    @property
    def model_class(self) -> str:
        return self.family_spec.model_class

    @property
    def experiment_name(self) -> str:
        return f"{self.family}_ablation_{self.key}_{self.dataset.lower()}"

    @property
    def slug(self) -> str:
        cleaned = re.sub(r"[^\w\-.]+", "_", self.experiment_name.strip())
        return re.sub(r"_+", "_", cleaned).strip("_")

    @property
    def baseline_experiment(self) -> str:
        """The published experiment to compare against, as ``<suite>/<DATASET>/<file>``."""
        return f"multi_round/{self.dataset}/{_benchmark_spec(self.family, self.dataset).slug}"

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
        note = self.family_spec.kernel_notes.get(self.first_kernel)
        if self.family == "cnn1d":
            blocks = "block" if self.depth == 1 else "blocks"
            suffix = f"; {note}" if note and self.first_kernel != 3 else ""
            return (
                f"{self.review_comment} ablation on {self.dataset}: {self.depth} conv "
                f"{blocks}, first kernel {self.first_kernel}, remaining kernels "
                f"{self.kernel}{suffix}"
            )

        if self.family == "alexnet":
            layers = "layer" if self.depth == 1 else "layers"
            shape = f"{self.depth} conv {layers}"
        else:
            per_stage = (self.depth - 2) // 8
            blocks = "BasicBlock" if per_stage == 1 else "BasicBlocks"
            shape = f"ResNet-{self.depth} ({per_stage} {blocks} per stage)"
        if self.study == "both":
            suffix = f"; the published {self.family_spec.label}"
        else:
            suffix = f"; {note}" if note else ""
        return (
            f"{self.review_comment} {self.family_spec.label} ablation on {self.dataset}: "
            f"{shape}, first kernel {self.first_kernel}{suffix}"
        )

    def as_dict(self) -> OrderedDict:
        return OrderedDict(
            family=self.family,
            variant=self.key,
            study=self.study,
            depth=self.depth,
            first_kernel=self.first_kernel,
            kernel=self.kernel,
        )


def variant_key(depth: int, first_kernel: int) -> str:
    return f"d{depth}_k{first_kernel}"


def family_variants(family: Family) -> List[Tuple[str, str, int, int]]:
    """``(key, study, depth, first_kernel)`` for the 7 distinct variants, in report order."""
    rows: List[Tuple[str, str, int, int]] = []
    for depth in family.depth_arm:
        study = "both" if depth == family.shared_depth else "depth"
        rows.append(
            (variant_key(depth, family.shared_kernel), study, depth, family.shared_kernel)
        )
    for first_kernel in family.kernel_arm:
        if first_kernel == family.shared_kernel:
            continue  # already emitted by the depth arm as the shared cell
        rows.append(
            (variant_key(family.shared_depth, first_kernel), "kernel", family.shared_depth,
             first_kernel)
        )
    return rows


def arm_variants(family: str, study: str) -> List[Tuple[str, float]]:
    """``(variant_key, level)`` in increasing-level order for one arm of one family."""
    spec = FAMILIES[family]
    if study == "depth":
        return [(variant_key(d, spec.shared_kernel), float(d)) for d in spec.depth_arm]
    return [(variant_key(spec.shared_depth, k), float(k)) for k in spec.kernel_arm]


#: Every variant key of any family, in registry order. The same key can name different
#: networks in different families (``d5_k3`` is a CNN1D and an AlexNet), so ``--variant``
#: is usually combined with ``--model``.
VARIANT_KEYS: Tuple[str, ...] = tuple(
    dict.fromkeys(row[0] for family in FAMILIES.values() for row in family_variants(family))
)

#: The kernel after the first, recorded in ``AblationSpec.kernel``. Only the 1D-CNN uses it
#: to build the model; ResNet's blocks are all 3 as published, and AlexNet keeps its
#: published 5-3-3-3 (recorded per variant in ``architecture.kernels_per_block``).
LATER_KERNEL: Mapping[str, Optional[int]] = {"cnn1d": 3, "alexnet": None, "resnet18": 3}


def build_registry() -> List[AblationSpec]:
    specs: List[AblationSpec] = []
    for dataset in DATASET_NAMES:
        for family in FAMILIES.values():
            published = _benchmark_spec(family.key, dataset)
            for key, study, depth, first_kernel in family_variants(family):
                specs.append(
                    AblationSpec(
                        family=family.key,
                        key=key,
                        dataset=dataset,
                        study=study,
                        depth=depth,
                        first_kernel=first_kernel,
                        kernel=LATER_KERNEL[family.key],
                        batch_size=published.batch_size,
                        lr=published.lr,
                        num_epochs=published.num_epochs,
                    )
                )
    return specs


REGISTRY: List[AblationSpec] = build_registry()


def select(
    datasets: Sequence[str] = DATASET_NAMES,
    keys: Sequence[str] = VARIANT_KEYS,
    studies: Optional[Sequence[str]] = None,
    families: Sequence[str] = FAMILY_KEYS,
) -> Iterator[AblationSpec]:
    """Yield the selected variants in registry order.

    Selecting by ``study`` includes each family's shared cell in either arm.
    """
    for spec in REGISTRY:
        if (
            spec.dataset not in datasets
            or spec.family not in families
            or spec.key not in keys
        ):
            continue
        if studies is not None and not any(
            (study == "depth" and spec.in_depth_arm)
            or (study == "kernel" and spec.in_kernel_arm)
            for study in studies
        ):
            continue
        yield spec


# ------------------------------------------------------------------------- document
def protocol_note(spec: AblationSpec) -> str:
    note = (
        "Trained with src.experiment.DeepLearningExperiment unchanged, i.e. the benchmark's "
        "own protocol: 20% of each training fold held out for val_loss logging only, no early "
        "stopping and no checkpoint selection (the tested model is the last epoch's). This "
        "deviates from the 10%-split-with-early-stopping described in Section 2.2 of the "
        "manuscript, and was chosen so these numbers remain directly comparable with the "
        f"already-published {spec.family_spec.label} row of Table 4."
    )
    if spec.family != "cnn1d":
        note += (
            " Batch size, learning rate and epochs are that row's, read from "
            "src/experiments.json, so they can differ between datasets (ResNet-18 on PU "
            "trains with batch 128 for 25 epochs)."
        )
    return note


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
    doc["family"] = spec.family
    doc["study"] = spec.study
    doc["variant"] = spec.key
    doc["description"] = spec.description
    doc["status"] = "executed"
    doc["source"] = OrderedDict(
        review_comment=spec.review_comment,
        review_file="review/reviwers_comments.md",
        baseline_experiment=spec.baseline_experiment,
        produced_by="run_experiments_ablation.py",
    )
    doc["run"] = run_info
    doc["architecture"] = architecture
    doc["configuration"] = configuration
    doc["results"] = results
    if fold_design:
        doc["fold_design"] = fold_design
    doc["notes"] = OrderedDict(
        protocol=protocol_note(spec), structure=spec.family_spec.structure_note
    )
    return doc


def document_family(doc: dict) -> Optional[str]:
    """The family of a result document; CNN1D results written before ``family`` existed
    carry it only in ``suite`` (``cnn1d_ablation``)."""
    family = doc.get("family")
    if family:
        return family
    suite = doc.get("suite") or ""
    if suite.endswith("_ablation"):
        return suite[: -len("_ablation")]
    return None
