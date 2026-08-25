"""Dataset loading and grouping, ported from the notebooks.

Each notebook builds its :class:`~vibdata.deep.DeepDataset.DeepDataset` the same way:
download the raw dataset, apply ``Sequential([SplitSampleRate()])``, optionally filter by
sample rate, and then assign every sample to a group with a dataset-specific
:class:`~signalAI.utils.group_dataset.GroupDataset` subclass.

Differences from the notebooks are marked with ``# FIX:`` and listed in
``scripts/README.md``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

import vibdata.raw as raw_datasets
from vibdata.deep.DeepDataset import DeepDataset, convertDataset, resample_dataset
from vibdata.deep.signal.core import SignalSample
from vibdata.deep.signal.transforms import FilterByValue, Sequential, SplitSampleRate

from signalAI.utils.group_dataset import GroupDataset


# --------------------------------------------------------------------------- grouping
class GroupCWRULoad(GroupDataset):
    @staticmethod
    def _assigne_group(sample: SignalSample) -> int:
        return sample["metainfo"]["load"]


class GroupMultiRoundCWRULoad(GroupDataset):
    @staticmethod
    def _assigne_group(sample: SignalSample) -> int:
        sample_metainfo = sample["metainfo"]
        return sample_metainfo["label"].astype(str) + " " + sample_metainfo["load"].astype(int).astype(str)


class GroupIMS(GroupDataset):
    NUM_FOLDS = 3

    def __init__(self, dataset, custom_name: str = None, **kwargs) -> None:
        super().__init__(dataset, custom_name, shuffle=True, **kwargs)
        keys = dataset.get_labels()
        values = dataset.get_labels_name()
        self.labels_name = dict(zip(keys, values))
        name_to_label = dict(zip(values, keys))
        metainfo = dataset.get_metainfo()
        defects_frequency = metainfo[metainfo.label != name_to_label["Normal"]].label.value_counts()
        self.defects_bins = {
            label: {"samples_per_fold": np.ceil(total / GroupIMS.NUM_FOLDS), "current_amount": 0}
            for label, total in defects_frequency.items()
        }

    def _get_group_divided(self, label: int):
        current_amount = self.defects_bins[label]["current_amount"]
        samples_per_fold = self.defects_bins[label]["samples_per_fold"]
        group = (current_amount // samples_per_fold) + 1
        self.defects_bins[label]["current_amount"] += 1
        return int(group - 1)

    def _assigne_group(self, sample: SignalSample) -> int:
        bearing = sample["metainfo"]["bearing"]
        test = sample["metainfo"]["test"]
        label = sample["metainfo"]["label"]
        label_str = self.labels_name[label]
        if test == 1 and bearing == 3:
            return 0 if label_str == "Normal" else self._get_group_divided(label)
        elif test == 1 and bearing == 4:
            return 1 if label_str == "Normal" else self._get_group_divided(label)
        elif test == 2 and bearing == 1:
            return 2 if label_str == "Normal" else self._get_group_divided(label)
        else:
            raise Exception(
                "Unexpected sample. The sample received is one of the conditions left out.\n"
                "The sample is of test: " + str(test) + " and bearing: " + str(bearing)
            )


class GroupMFPT(GroupDataset):
    NUM_FOLDS = 3
    FAKE_OUTER_RACE_270_LABEL = 100

    def __init__(self, dataset: DeepDataset, custom_name: str = None, **kwargs) -> None:
        super().__init__(dataset, custom_name, shuffle=True, **kwargs)
        keys = dataset.get_labels()
        values = dataset.get_labels_name()
        self.labels_name = dict(zip(keys, values))
        name_to_label = dict(zip(values, keys))
        metainfo = dataset.get_metainfo().copy()
        outer_race_270_mask = (metainfo.label == name_to_label["Outer Race"]) & (metainfo.load == 270)
        metainfo.loc[outer_race_270_mask, "label"] = GroupMFPT.FAKE_OUTER_RACE_270_LABEL
        labels_frequency = metainfo.label.value_counts()
        self.labels_bins = {
            label: {"samples_per_fold": np.ceil(total / GroupMFPT.NUM_FOLDS), "current_amount": 0}
            for label, total in labels_frequency.items()
        }

    def _get_group_divided(self, label):
        current_amount = self.labels_bins[label]["current_amount"]
        samples_per_fold = self.labels_bins[label]["samples_per_fold"]
        group = (current_amount // samples_per_fold) + 1
        self.labels_bins[label]["current_amount"] += 1
        return int(group - 1)

    def _assigne_group(self, sample: SignalSample) -> int:
        label = sample["metainfo"]["label"]
        label_str = self.labels_name[label]
        load = sample["metainfo"]["load"]
        if label_str == "Outer Race" and load == 270:
            label = self.FAKE_OUTER_RACE_270_LABEL
        return self._get_group_divided(label)


class GroupMultiRoundMFPT(GroupDataset):
    @staticmethod
    def _assigne_group(sample: SignalSample) -> int:
        sample_metainfo = sample["metainfo"]
        return sample_metainfo["label"].astype(str) + " " + sample_metainfo["load"].astype(int).astype(str)


class GroupPU(GroupDataset):
    @staticmethod
    def _assigne_group(sample: SignalSample) -> int:
        rotation_speed = sample["metainfo"]["file_name"][:3]
        load_torque = sample["metainfo"]["load_nm"]
        radial_force = sample["metainfo"]["radial_force_n"]
        if rotation_speed == "N15" and load_torque == 0.7 and radial_force == 1000:
            return 0
        elif rotation_speed == "N09" and load_torque == 0.7 and radial_force == 1000:
            return 1
        elif rotation_speed == "N15" and load_torque == 0.1 and radial_force == 1000:
            return 2
        elif rotation_speed == "N15" and load_torque == 0.7 and radial_force == 400:
            return 3
        else:
            raise Exception("Unexpected operating condition")


class GroupMultiRoundPU(GroupDataset):
    @staticmethod
    def _assigne_group(sample: SignalSample) -> int:
        sample_metainfo = sample["metainfo"]
        condition_fields = ["radial_force_n", "rotation_hz", "load_nm"]
        condition_str = "_".join([sample_metainfo[field].astype(str) for field in condition_fields])
        return sample_metainfo["label"].astype(str) + " " + condition_str


class GroupUOC(GroupDataset):
    NUM_FOLDS = 5

    def __init__(self, dataset, custom_name: str = None, **kwargs) -> None:
        super().__init__(dataset, custom_name, shuffle=True, **kwargs)
        keys = dataset.get_labels()
        values = dataset.get_labels_name()
        self.labels_name = dict(zip(keys, values))
        self.labels_bins = {label: {fold: 0 for fold in range(1, GroupUOC.NUM_FOLDS + 1)} for label in keys}

    def _assigne_group(self, sample: SignalSample) -> int:
        severity = sample["metainfo"]["severity"]
        if severity != "-":
            return int(severity) - 1
        else:
            label = sample["metainfo"]["label"]
            group = min(self.labels_bins[label], key=self.labels_bins[label].get)
            self.labels_bins[label][group] += 1
            return group - 1


# ------------------------------------------------------------------------- resampling
class ResamplerIMS:
    """Under-samples the ``Normal`` class of IMS down to the number of defect samples."""

    def resample(self, dataset: DeepDataset) -> DeepDataset:
        # Get metainfo from complete dataset
        metainfo = dataset.get_metainfo()

        # All the defects signals are kept
        defect_mask = metainfo.label != 6
        new_defects = metainfo[defect_mask]

        # Number of samples to be resampling of the class `Normal` (label = 6)
        # Is the same number of samples of the defects classes
        normals_number = new_defects.shape[0]

        # Resample the normals labels
        normals_mask = [
            (metainfo.test == 1) & (metainfo.bearing == 4) & (metainfo.label == 6),
            (metainfo.test == 1) & (metainfo.bearing == 3) & (metainfo.label == 6),
            (metainfo.test == 2) & (metainfo.bearing == 1) & (metainfo.label == 6),
        ]
        all_normals = pd.concat([metainfo[mask] for mask in normals_mask]).sort_index()
        new_normals = self._resample_normals(all_normals, normals_number)

        # Concat the results in order to compute the new results
        resampled_metainfo = pd.concat([new_normals, new_defects])
        # Get the indexes of the samples that will be kept and ensure that they are sorted
        new_indexes = resampled_metainfo.index.values
        new_indexes.sort()
        resampled_dataset = resample_dataset(dataset, new_indexes)
        return resampled_dataset

    @staticmethod
    def _resample_normals(normals: pd.DataFrame, normals_number: int) -> pd.DataFrame:
        from imblearn.under_sampling import RandomUnderSampler

        # Compute column to indetify each condition
        normals = normals.copy()
        normals["set"] = normals["test"] + normals["bearing"]

        def compute_new_frequency(set: int):
            old_freq = sum(normals["set"] == set) / normals.shape[0]
            return np.ceil(normals_number * old_freq).astype("int32")

        # Compute the new frequency for each condition
        sampling_strategy = {set: compute_new_frequency(set) for set in normals.set.unique()}

        rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
        y = normals.set.values.astype("int32")
        new_normals, _ = rus.fit_resample(normals, y)
        new_normals.drop(columns=["set"], inplace=True)

        return new_normals


# ---------------------------------------------------------------------------- registry
@dataclass(frozen=True)
class DatasetSpec:
    """Everything needed to rebuild one notebook's ``deep_dataset_time``."""

    name: str
    raw_class: str                      # attribute of ``vibdata.raw``
    raw_subdir: str                     # folder under ``<data-root>/raw_data``
    sample_rate: Optional[int] = None   # ``FilterByValue`` on ``sample_rate``; None = no filter
    resampler: Optional[Callable[[], object]] = None

    # single round grouping
    group_class: type = None
    group_name: str = ""                # ``dataset_name`` handed to FoldIdxGeneratorUnbiased

    # multi round grouping (None when the notebook has no true multiround run)
    multiround_group_class: Optional[type] = None
    multiround_group_name: str = ""
    class_def: Dict[int, str] = field(default_factory=dict)
    condition_def: Dict[str, str] = field(default_factory=dict)

    @property
    def has_multiround(self) -> bool:
        return self.multiround_group_class is not None


DATASETS: Dict[str, DatasetSpec] = {
    "CWRU12k": DatasetSpec(
        name="CWRU12k",
        raw_class="CWRU_raw",
        raw_subdir="cwru",
        sample_rate=12000,
        group_class=GroupCWRULoad,
        group_name="CWRU12k_single",
        multiround_group_class=GroupMultiRoundCWRULoad,
        multiround_group_name="CWRU12k_multi",
        class_def={0: "N", 1: "O", 2: "I", 3: "R"},
        condition_def={"0": "0", "1": "1", "2": "2", "3": "3"},
    ),
    "CWRU48k": DatasetSpec(
        name="CWRU48k",
        raw_class="CWRU_raw",
        raw_subdir="cwru",
        sample_rate=48000,
        group_class=GroupCWRULoad,
        group_name="CWRU48k_deep",
        multiround_group_class=GroupMultiRoundCWRULoad,
        multiround_group_name="CWRU48k_multi",
        class_def={0: "N", 1: "O", 2: "I", 3: "R"},
        condition_def={"0": "0", "1": "1", "2": "2", "3": "3"},
    ),
    "IMS": DatasetSpec(
        name="IMS",
        raw_class="IMS_raw",
        raw_subdir="ims",
        sample_rate=None,
        resampler=ResamplerIMS,
        group_class=GroupIMS,
        group_name="IMS_deep",
        # The multi_round notebook for IMS still uses ``folds_singleround_deep``:
        # there is no multiround grouping to port.
    ),
    "MFPT": DatasetSpec(
        name="MFPT",
        raw_class="MFPT_raw",
        raw_subdir="mfpt",
        sample_rate=48828,
        group_class=GroupMFPT,
        group_name="MFPT_deep",
        multiround_group_class=GroupMultiRoundMFPT,
        multiround_group_name="MFPT_multi",
        class_def={23: "N", 25: "O", 24: "I"},
        condition_def={
            "0": "C1", "25": "C2", "50": "C3", "100": "C4",
            "150": "C5", "200": "C6", "250": "C7", "300": "C8",
        },
    ),
    "PU": DatasetSpec(
        name="PU",
        raw_class="PU_raw",
        raw_subdir="pu",
        sample_rate=None,
        group_class=GroupPU,
        group_name="PU_deep",
        multiround_group_class=GroupMultiRoundPU,
        multiround_group_name="PU_multi",
        class_def={26: "N", 27: "O", 28: "I", 29: "R"},
        condition_def={
            "1000_15.0_0.7": "0", "1000_25.0_0.1": "1",
            "1000_25.0_0.7": "2", "400_25.0_0.7": "3",
        },
    ),
    "UOC": DatasetSpec(
        name="UOC",
        raw_class="UOC_raw",
        raw_subdir="uoc",
        sample_rate=None,
        group_class=GroupUOC,
        group_name="UOC_deep",
        # Same as IMS: the multi_round notebook runs a single round.
    ),
}

DATASET_NAMES = tuple(DATASETS)


def load_deep_dataset(spec: DatasetSpec, data_root: str, download: bool = True) -> DeepDataset:
    """Rebuild the notebook's ``deep_dataset_time`` for ``spec``.

    FIX (vs. notebooks): every notebook converts into the same
    ``../data/deep_data/deep_learning`` directory. ``convertDataset`` hashes only the raw
    class, its length and the transforms -- not the ``filter`` -- so reusing one directory
    either raises "Dataset corrupted!" or silently returns another dataset's cache
    (CWRU12k vs CWRU48k differ only by the filter). Here each dataset gets its own
    directory under ``<data-root>/deep_data/<name>``.
    """
    raw_root_dir = os.path.join(data_root, "raw_data", spec.raw_subdir)
    raw_dataset = getattr(raw_datasets, spec.raw_class)(raw_root_dir, download=download)

    transforms_time = Sequential([SplitSampleRate()])
    filter_ = None
    if spec.sample_rate is not None:
        filter_ = FilterByValue(on_field="sample_rate", values=spec.sample_rate)

    deep_root_dir = os.path.join(data_root, "deep_data", spec.name)
    deep_dataset = convertDataset(
        raw_dataset,
        filter=filter_,
        transforms=transforms_time,
        dir_path=deep_root_dir,
        batch_size=32,
    )
    if spec.resampler is not None:
        deep_dataset = spec.resampler().resample(deep_dataset)
    return deep_dataset
