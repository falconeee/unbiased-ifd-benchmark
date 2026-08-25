"""The 108 experiments of the benchmark: 2 suites x 6 datasets x 9 models.

``experiments.json`` (next to this module) was generated from the JSONs under ``v0_experiments/v0_results/`` so that experiment
names, descriptions and hyperparameters match the notebooks exactly (they are not uniform:
``pretrain_epochs`` is 50 for some datasets and 100 for others, and ``multi_round/PU``
trains ResNet18 with ``batch_size=128`` for 25 epochs).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence

EXPERIMENTS_PATH = os.path.join(os.path.dirname(__file__), "experiments.json")

SUITES = ("single_round", "multi_round")
MODEL_KEYS = ("mlp1d", "ae1d", "sae1d", "dae1d", "cnn1d", "lenet1d", "resnet18", "alexnet", "bilstm")

#: Models whose training has an autoencoder pre-training phase.
AUTOENCODER_KEYS = ("ae1d", "sae1d", "dae1d")


@dataclass(frozen=True)
class ExperimentSpec:
    suite: str
    dataset: str
    key: str
    experiment_name: str
    description: str
    model_class: str
    protocol: str
    section: Optional[str]
    batch_size: int
    lr: float
    num_epochs: int
    pretrain_epochs: int
    recon_loss_weight: Optional[float]
    sparsity_target: Optional[float]
    sparsity_weight: Optional[float]
    notebook_status: str
    notebook_experiment_name_template: Optional[str]

    @property
    def is_autoencoder(self) -> bool:
        return self.key in AUTOENCODER_KEYS

    @property
    def training_stage(self) -> str:
        if self.is_autoencoder and self.pretrain_epochs > 0:
            return "autoencoder+classifier"
        return "classifier"

    @property
    def slug(self) -> str:
        """File name used under ``results/<suite>/<dataset>/`` (matches the v0 results)."""
        cleaned = re.sub(r"[^\w\-.]+", "_", self.experiment_name.strip())
        return re.sub(r"_+", "_", cleaned).strip("_")

    def as_dict(self) -> dict:
        return {
            "experiment_name": self.experiment_name,
            "description": self.description,
            "section": self.section,
            "notebook_experiment_name_template": self.notebook_experiment_name_template,
        }


def _load() -> Dict[str, List[ExperimentSpec]]:
    with open(EXPERIMENTS_PATH, encoding="utf-8") as fh:
        raw = json.load(fh)
    table: Dict[str, List[ExperimentSpec]] = {}
    for notebook_key, entries in raw.items():
        suite, dataset = notebook_key.split("/")
        table[notebook_key] = [
            ExperimentSpec(suite=suite, dataset=dataset, **entry) for entry in entries
        ]
    return table


REGISTRY: Dict[str, List[ExperimentSpec]] = _load()

DATASET_NAMES = tuple(dict.fromkeys(key.split("/")[1] for key in REGISTRY))


def select(
    suites: Sequence[str] = SUITES,
    datasets: Sequence[str] = DATASET_NAMES,
    keys: Sequence[str] = MODEL_KEYS,
) -> Iterator[ExperimentSpec]:
    """Yield the selected experiments grouped by notebook, in registry order."""
    for suite in suites:
        for dataset in datasets:
            notebook_key = f"{suite}/{dataset}"
            for spec in REGISTRY.get(notebook_key, []):
                if spec.key in keys:
                    yield spec
