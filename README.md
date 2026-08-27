# unbiased-ifd-benchmark

Data and code behind the paper: a benchmark for **intelligent fault diagnosis** (IFD) on
rolling-bearing vibration signals, evaluated under an **unbiased** protocol. Instead of
splitting samples at random, every test fold holds out whole acquisition groups — motor load
for CWRU, operating condition for PU, bearing and test rig for IMS, fault severity for UOC. A
model therefore has to generalise to a condition it never saw, rather than to another window of
a recording it already memorised.

Nine 1D deep-learning architectures across six public datasets, in two protocols —
**108 experiments** in total.

## Findings

Mean test accuracy over the unbiased folds. Numbers come from the first round of experiments,
archived in `v0_experiments/v0_results/`; the class count is in the header because it sets the
chance level for each dataset.

**Single round**

| Model | CWRU12k (4) | CWRU48k (3) | IMS (7) | MFPT (2) | PU (4) | UOC (5) |
|---|---|---|---|---|---|---|
| MLP1D | 0.438 | 0.374 | 0.161 | 0.500 | — | 0.924 |
| AE1D | 0.438 | 0.344 | 0.191 | 0.500 | — | 0.922 |
| SAE1D | 0.457 | 0.350 | 0.155 | 0.500 | — | 0.873 |
| DAE1D | 0.444 | 0.354 | 0.180 | 0.500 | — | 0.833 |
| CNN1D | 0.877 | 0.859 | 0.216 | 0.929 | — | 0.966 |
| LeNet1D | 0.806 | 0.851 | 0.143 | 0.929 | — | 0.805 |
| ResNet18 | 0.927 | 0.742 | 0.201 | 1.000 | — | 0.975 |
| AlexNet1D | 0.856 | 0.777 | 0.143 | 0.976 | — | 0.941 |
| BiLSTM | 0.889 | 0.827 | 0.243 | 1.000 | — | 0.944 |

**Multi round** (the condition-to-fold mapping is rotated and the whole cross validation
repeated, so the score does not hang on one lucky arrangement of conditions)

| Model | CWRU12k (4) | CWRU48k (3) | IMS (7) | MFPT (2) | PU (4) | UOC (5) |
|---|---|---|---|---|---|---|
| MLP1D | 0.416 | 0.358 | 0.161 | 0.500 | — | 0.924 |
| AE1D | 0.427 | 0.350 | 0.191 | 0.500 | — | 0.922 |
| SAE1D | 0.430 | 0.355 | 0.155 | 0.500 | — | 0.873 |
| DAE1D | 0.401 | 0.349 | 0.180 | 0.500 | — | 0.833 |
| CNN1D | 0.879 | 0.866 | 0.216 | 0.900 | 0.716 | 0.966 |
| LeNet1D | 0.810 | 0.811 | 0.143 | 0.919 | 0.715 | 0.805 |
| ResNet18 | 0.894 | 0.825 | 0.201 | 1.000 | 0.627 | 0.975 |
| AlexNet1D | 0.816 | 0.713 | 0.143 | 0.848 | 0.723 | 0.941 |
| BiLSTM | 0.825 | 0.872 | 0.243 | 0.929 | 0.704 | 0.944 |

Thirteen cells are empty: the nine single-round PU experiments and the four multi-round PU
autoencoder experiments were never executed in the archived notebooks. IMS and UOC have no
multi-round design, so their multi-round entries repeat the single-round result (see
[Fold design](#fold-design)).

What the tables say:

- **Group-held-out scores are far below the near-perfect accuracies these datasets usually
  report.** The gap is the size of the bias that random splitting hides.
- **The dense and autoencoder family collapses to roughly chance** on the harder datasets:
  0.40–0.46 on CWRU12k (4 classes), 0.34–0.37 on CWRU48k (3 classes), 0.500 on MFPT
  (2 classes). Whatever MLP1D, AE1D, SAE1D and DAE1D learn is tied to the acquisition
  condition, not to the fault. Unsupervised pre-training — plain, sparse or denoising — does
  not change that.
- **Convolutional and recurrent models do transfer across conditions**, but unevenly: CNN1D and
  ResNet18 reach 0.88–0.93 on CWRU12k while ResNet18 drops to 0.74 on CWRU48k, and PU sits at
  0.63–0.72 for every architecture that ran it.
- **IMS is at chance for every architecture** (0.14–0.24 over 7 classes). Holding out the test
  rig and bearing leaves nothing transferable in this feature space.
- **Multi round shifts the numbers by a few points but not the ranking**, and mostly downward
  for the weaker models — the single-round figure tends to be the optimistic one.
- **Dataset difficulty is not what the literature ordering suggests.** UOC stays easy for every
  architecture (0.80–0.98) and MFPT for every architecture that learns anything at all
  (0.85–1.00), CWRU degrades, PU is hard, IMS is unsolved.

Per-fold accuracy, F1, precision, recall, ROC AUC, confusion matrices and full training curves
for every experiment are in the JSON files described under [Results](#results).

---

## Layout

```
run_experiments.py           entry point: the 108 benchmark experiments
run_experiments_ablation.py  entry point: the 21 ablation experiments (see Ablations)

src/                         the benchmark, shared by both entry points
├── registry.py              the 108 experiments (+ experiments.json)
├── datasets.py              the 6 datasets: download, transform, grouping, resampling
├── folds.py                 unbiased folds, single and multi round (+ fold_designs.json)
├── models.py                the 9 architectures
├── experiment.py            DeepLearningExperiment: cross validation and training loop
└── serialize.py             results -> JSON

ablation/                    only what the ablations add on top of src/
├── models.py                parametric CNN1D: depth and initial kernel as arguments
├── registry.py              the 21 ablation experiments + their JSON document
└── report.py                ablation results -> tables and significance tests

results/                     output of a benchmark run (created on first run)
results_ablation/            output of an ablation run (created on first run)
data/                        raw downloads and converted datasets (git-ignored)
review/                      reviewer comments driving the ablations
paper/                       the manuscript
v0_experiments/              archived first version: the original notebooks and their results
requirements.txt
```

`ablation/` reuses `src/` untouched — same datasets, same folds, same training loop, same
result serialisation — so ablation numbers are directly comparable with the published ones.

The benchmark started as twelve Colab notebooks, archived unchanged in `v0_experiments/`.
`run_experiments.py` is their script form: same datasets, same folds, same architectures, same
hyperparameters (which are *not* uniform — `pretrain_epochs` is 50 for some datasets and 100
for others, and `multi_round/PU` trains ResNet18 with `batch_size=128` for 25 epochs).
`v0_experiments/v0_results/` holds the results parsed out of the notebook cell outputs, in the
same JSON schema the script writes, so old and new runs can be compared field by field.

---

## Install

[uv](https://docs.astral.sh/uv/) is the recommended way — resolving and installing torch and
the rest takes seconds instead of minutes. From the repository root:

```bash
# If you don't have uv installed
pip install uv

# Create the virtual environment (the repo has no pyproject.toml, hence --no-project)
uv venv --no-project

# Activate the virtual environment
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows

# Install project requirements using uv and requirements.txt
uv pip install -r requirements.txt
```

<details>
<summary>Without uv</summary>

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
</details>

`gdown<5` is not optional: `vibdata==1.1.1` calls `gdown.cached_download(..., md5=...)`, and
gdown 5 removed that argument, so every raw download fails with
`TypeError: download() got an unexpected keyword argument 'md5'`.

Validated on Python 3.12 with torch 2.12 and 2.13, on CPU.

## Run

There are two entry points. They share the same flag vocabulary and the same result schema:

| Script | Runs | Writes to |
|---|---|---|
| `run_experiments.py` | the 108 benchmark experiments | `results/` |
| `run_experiments_ablation.py` | the 21 ablation experiments | `results_ablation/` |

Both read and write the same `data/` directory, so a dataset downloaded by one is reused by
the other. This section documents the benchmark runner; the ablation runner has its own
section under [Ablations](#ablations).

With the environment activated:

```bash
python run_experiments.py --all --resume --keep-going
```

`uv run --no-project python run_experiments.py ...` works too, and uses `./.venv` without
activating anything.

### Choosing what runs

Three selection filters, all repeatable. Passing a filter more than once *widens* the
selection along that dimension; different filters *narrow* each other:

| Filter | Accepted values |
|---|---|
| `--suite` | `single_round`, `multi_round` |
| `--dataset` | `CWRU12k`, `CWRU48k`, `IMS`, `MFPT`, `PU`, `UOC` |
| `--model` | `mlp1d`, `ae1d`, `sae1d`, `dae1d`, `cnn1d`, `lenet1d`, `resnet18`, `alexnet`, `bilstm` |

Passing no filter is the same as `--all`. `--list` prints the resulting selection and exits
without training — always worth running first:

```bash
# 108 rows: everything
python run_experiments.py --all --list

# 18 rows: two models, every dataset, both protocols
python run_experiments.py --model resnet18 --model alexnet --list
```

### Commands you will actually use

```bash
# The full grid, resumable and fault-tolerant. This is the main command.
#   --resume     skips experiments whose result JSON already exists
#   --keep-going carries on to the next experiment when one raises
python run_experiments.py --all --resume --keep-going

# The same thing detached, keeping a transcript. Results are written per experiment,
# so you can tail run.out or results/run.log and stop it at any point.
nohup python run_experiments.py --all --resume --keep-going > run.out 2>&1 &

# One notebook's worth of work: nine models on one dataset under one protocol.
# The dataset is loaded and converted once and shared by all nine.
python run_experiments.py --suite single_round --dataset CWRU12k

# A single experiment, cut down so it finishes in seconds. Use this to check the
# environment before committing to a long run; the overrides are recorded in the
# result JSON under run.overrides so a smoke result is never mistaken for a real one.
python run_experiments.py --suite single_round --dataset UOC \
    --model cnn1d --epochs 2 --max-rounds 1

# Fill in only what is missing after an interrupted run, without re-reading anything.
python run_experiments.py --all --resume --no-download
```

Raw datasets are downloaded on first use into `data/` and converted once per dataset, so the
first run touching a given dataset is much slower than the rest.

### Flags

**Selection**

| Flag | Default | Notes |
|---|---|---|
| `--all` | — | run everything; same as passing no filter |
| `--suite`, `--dataset`, `--model` | all | repeatable selection filters (see above) |
| `--list` | off | print the selection and exit without training |

**Paths**

| Flag | Default | Notes |
|---|---|---|
| `--data-root` | `data/` | raw downloads, converted datasets, group/fold caches |
| `--output-dir` | `results/` | result JSONs, `index.json`, `run_manifest.json` |
| `--artifacts-dir` | `<output-dir>/_artifacts` | per-fold loss curves and `.pt` checkpoints |
| `--log-file` | `<output-dir>/run.log` | full transcript, appended |
| `--no-download` | off | fail instead of downloading a missing dataset |

**Execution**

| Flag | Default | Notes |
|---|---|---|
| `--device` | `cuda` if available | any torch device string |
| `--seed` | `42` | seeds python/numpy/torch, the validation split and the weight init |
| `--resume` | off | skip experiments whose result JSON already exists |
| `--keep-going` | off | carry on when one experiment raises; failures land in `run_manifest.json` |
| `--no-artifacts` | off | skip loss curves and checkpoints (much less disk) |
| `--folds-source` | `notebook` | `notebook` rebuilds the folds from the recorded design; `generate` runs the real combination search — see [Fold design](#fold-design) |

**Overrides** — for smoke tests only. Each one is recorded in the result JSON under
`run.overrides`, so a shortened run is always identifiable.

| Flag | Overrides |
|---|---|
| `--epochs` | `num_epochs` |
| `--pretrain-epochs` | `pretrain_epochs` (autoencoders only) |
| `--batch-size` | `batch_size` |
| `--max-rounds` | number of rounds executed in a multiround experiment |

### Resuming, logs and failures

Progress goes to stdout **and** to `<output-dir>/run.log`, so a backgrounded run keeps a full
transcript. Results are written per experiment, not at the end, so `--resume` picks up exactly
where it stopped — including after a crash or a `Ctrl-C`. Every run also writes
`run_manifest.json` listing what was executed, skipped and failed, plus the exact command,
seed, device and package versions.

The exit code is `1` if any experiment failed, `0` otherwise.

## Results

```
results/
├── index.json                                    every experiment, one row each
├── run_manifest.json                             what this run did: args, env, failures
├── run.log
├── _artifacts/<suite>/<DATASET>/<experiment>/    loss curves + model checkpoints
└── <suite>/<DATASET>/
    ├── _index.json
    └── <experiment_name>.json
```

Each experiment JSON:

| Key | Contents |
|---|---|
| `experiment_name`, `dataset`, `suite`, `protocol`, `model`, `description` | identity |
| `source` | the v0 notebook it corresponds to |
| `run` | timestamps, duration, seed, device, fold source, overrides, package versions |
| `configuration` | `batch_size`, `lr`, `num_epochs`, `pretrain_epochs`, sparsity/reconstruction settings, `input_length`, `num_classes` |
| `results.summary` | mean/std of accuracy and F1 |
| `results.per_fold_metrics` | accuracy, F1, precision, recall, roc_auc per fold |
| `results.confusion_matrix` | pooled matrix, overall accuracy, sample count |
| `results.folds_training_log` | per fold: per-fold confusion matrix and **every** epoch (`train_loss`, `val_loss`, `time_seconds`; plus `recon_loss` for the autoencoders) |
| `results.fold_errors` | present only when a fold failed |

Multiround experiments replace `folds_training_log` with `rounds[]`, each round carrying its
own summary, per-fold metrics, confusion matrix and training logs, plus a `fold_design`
describing the round x fold group assignment.

`v0_experiments/v0_results/` uses the same schema, so a new run can be diffed against the
notebook numbers directly. The v0 files have no `run` block and their training history only has
every fifth epoch — that is all the notebooks printed.

Two things to know when reading a number: per-fold `accuracy` is `balanced_accuracy_score`,
while the pooled "overall accuracy" of the confusion matrix is the raw one, so the two differ;
and there is no early stopping or checkpoint selection — the model tested is the one from the
last epoch.

---

## Ablations

`run_experiments_ablation.py` answers Reviewer #2's two objections (`review/reviwers_comments.md`):
that the AlexNet vs. ResNet-18 comparison cannot isolate depth, because the two differ in
kernels, receptive fields, pooling, parameter count and skip connections at once; and that the
receptive-field explanation was never tested. Both are answered by varying **one** axis at a
time on the paper's own 1D-CNN.

### The grid

Variants are named `d<depth>_k<first kernel>`. Every kernel after the first is 3, channel
widths are 16→32→64→128→256, and padding, BatchNorm, pooling strategy, dropout and the
classification head are identical throughout.

| Arm | Comment | Variants |
|---|---|---|
| depth | R2.2 | `d1_k3`, `d2_k3`, **`d3_k3`**, `d5_k3` |
| receptive field | R2.3 | **`d3_k3`**, `d3_k7`, `d3_k11`, `d3_k64` |

`d3_k3` belongs to both arms, so the grid is 7 distinct variants, not 8 — it is trained once
and reported in both tables. 7 variants × 3 datasets (PU, CWRU12k, CWRU48k) = **21 experiments**,
each 8 rounds × 4 folds = 672 fold trainings. `d3_k3` is `src/models.py`'s `CNN1D` with its
7-5-3 kernel schedule flattened to 3-3-3; block structure is otherwise identical, including the
`AdaptiveMaxPool1d(16)` that replaces the last block's `MaxPool1d(2)`.

The two arms differ in how cleanly they isolate their axis, and the result JSON records enough
to say so in the manuscript:

| Variant | Depth | First kernel | Params | Receptive field |
|---|---|---|---|---|
| `d1_k3` | 1 | 3 | 33,508 | 3 |
| `d2_k3` | 2 | 3 | 67,908 | 8 |
| `d3_k3` | 3 | 3 | 139,780 | 18 |
| `d5_k3` | 5 | 3 | 657,028 | 78 |
| `d3_k7` | 3 | 7 | 139,844 | 22 |
| `d3_k11` | 3 | 11 | 139,908 | 26 |
| `d3_k64` | 3 | 64 | 140,756 | 79 |

The receptive-field arm is close to a clean manipulation: kernel 3 → 64 moves the receptive
field 4.4× while parameter count changes by 0.7%. The depth arm is not, and cannot be — adding
a block adds its weights and widens the flattened head, so parameters grow 20× across it.
That confound is inherent to the design the response letter proposes; it is measured
(`architecture.num_parameters`) rather than hidden, and `d5_k3` vs `d3_k64` gives a useful
cross-arm reading: near-identical receptive field (78 vs 79), 4.7× the parameters.

### Training protocol

Identical to the published 1D-CNN row on each dataset — same unbiased multiround folds, Adam,
lr 3e-4, batch 64, 100 epochs — and the same `DeepLearningExperiment` training loop, so a 20%
validation split used only for logging and **no early stopping**. Note this differs from
Section 2.2 of the manuscript, which describes a 10% split with early stopping; the benchmark
code has never done that, and the ablation follows the code so its numbers stay comparable
with Table 4. Both facts are recorded in every result JSON under `configuration.val_split`,
`configuration.early_stopping` and `notes.protocol`.

### Selecting variants

Same idea as the benchmark runner, with `--model` replaced by two filters:

| Filter | Accepted values | Notes |
|---|---|---|
| `--dataset` | `PU`, `CWRU12k`, `CWRU48k` | repeatable |
| `--variant` | `d1_k3`, `d2_k3`, `d3_k3`, `d5_k3`, `d3_k7`, `d3_k11`, `d3_k64` | repeatable |
| `--study` | `depth`, `kernel` | selects a whole arm; **either arm includes the shared `d3_k3` cell** |

`--variant` and `--study` intersect, so `--study depth --variant d5_k3` runs just `d5_k3`.
Passing neither runs all seven.

### Commands

```bash
# The full grid: 21 experiments, 672 fold trainings. This is the main command.
python run_experiments_ablation.py --all --resume --keep-going

# Detached, with a transcript, for the long run.
nohup python run_experiments_ablation.py --all --resume --keep-going > run_ablation.out 2>&1 &

# Check the selection before spending compute. Prints dataset, variant, arm, depth
# and first kernel for each experiment, then exits.
python run_experiments_ablation.py --all --list

# Only the dataset the reviewer's objection is actually about (7 experiments).
# Start here if you want an answer before the full grid finishes.
python run_experiments_ablation.py --dataset PU

# One arm on one dataset (4 experiments, including the shared d3_k3 cell):
python run_experiments_ablation.py --dataset PU --study depth     # R2.2
python run_experiments_ablation.py --dataset PU --study kernel    # R2.3

# Smoke test: one variant, 2 epochs, 1 round. Finishes in seconds and proves the
# pipeline works end to end; the overrides are recorded in run.overrides.
python run_experiments_ablation.py --dataset CWRU12k --variant d1_k3 \
    --epochs 2 --max-rounds 1

# Rebuild report.md / report.tex / report.json from results already on disk,
# without training anything. Safe to run at any time, including mid-run.
python run_experiments_ablation.py --report
```

### Flags that differ

Everything from the benchmark runner's [Flags](#flags) applies unchanged — `--data-root`,
`--output-dir`, `--artifacts-dir`, `--log-file`, `--no-download`, `--device`, `--seed`,
`--resume`, `--keep-going`, `--no-artifacts`, `--folds-source`, `--epochs`, `--batch-size`,
`--max-rounds` — with two differences and one addition:

| Flag | Default | Notes |
|---|---|---|
| `--output-dir` | `results_ablation/` | not `results/`; keep them apart |
| `--epochs`, `--batch-size`, `--max-rounds` | — | the only overrides; there is no `--pretrain-epochs` because no ablation variant pre-trains |
| `--report` | off | rebuild the report from `--output-dir` and exit without training |

`--data-root` defaults to the same `data/` as the benchmark, so nothing is downloaded twice.

### Output

`results_ablation/` mirrors `results/` one level flatter (no suite directory), with an extra
top-level `architecture` block per experiment and `study`/`variant` fields:

```
results_ablation/
├── index.json                               every variant, one row each, with params and RF
├── run_manifest.json                        what this run did: args, env, failures
├── run.log
├── report.md                                tables + significance tests, for reading
├── report.tex                               the same tables as booktabs, for the manuscript
├── report.json                              the same numbers, machine readable
├── _artifacts/<DATASET>/<experiment>/       loss curves + model checkpoints
└── <DATASET>/
    ├── _index.json
    └── <experiment_name>.json
```

Each experiment JSON is the benchmark schema plus:

| Key | Contents |
|---|---|
| `study`, `variant` | which arm the variant belongs to (`depth`, `kernel` or `both`) and its name |
| `source` | the review comment it answers (`R2.2`, `R2.3`) and the baseline experiment to compare against |
| `architecture` | depth, kernels and channels per block, parameter counts (total, conv, head), theoretical receptive field in samples and seconds, flattened feature size |
| `configuration` | adds `optimizer`, `loss`, `val_split`, `early_stopping`, `checkpoint_selection` |
| `notes` | why the protocol and block structure are what they are |

### Reading the report

The report is regenerated at the end of every run, and by `--report` alone. Per dataset and
per arm it gives:

- **Mean ± std** of Balanced Accuracy and Macro F1, under *both* dispersion conventions — std
  over all 32 per-fold scores (what Table 4's caption claims) and std over the 8 round means
  (what `results.summary` stores) — because the two differ and the manuscript is ambiguous.
- **Adjacent paired Wilcoxon** tests between neighbouring variants, paired by `(round, fold)`
  and Holm-corrected within the arm. Pairing is legitimate because every variant sees the
  identical fold assignment with the identical seed.
- **Page's trend test** across the whole arm, run in both directions — this is the test that
  actually addresses "monotonic in depth", which adjacent pairwise tests do not.
- **Spearman ρ** against depth or kernel size, for effect direction and magnitude.

`report.tex` contains the same tables as `booktabs` environments with `\label{tab:ablation_<arm>_<dataset>}`,
ready to paste into the manuscript.

---

## Protocol

**Datasets and grouping.** Every test fold holds out whole acquisition groups:

| Dataset | Classes | Grouped by | Folds |
|---|---|---|---|
| CWRU 12k | 4 | motor load (0–3 hp) | 4 |
| CWRU 48k | 3 | motor load (0–3 hp) | 4 |
| IMS | 7 | test rig and bearing | 3 |
| MFPT | 2 | fault type and load, binned | 3 (single round) / 7 (multi round) |
| PU | 4 | rotation speed, load torque, radial force | 4 |
| UOC | 5 | fault severity | 5 |

**Architectures**, four of them with an unsupervised pre-training phase:

| Model key | Class | Pre-training |
|---|---|---|
| `mlp1d` | `MLP1D` | — |
| `ae1d` | `AE1D` | autoencoder |
| `sae1d` | `SAE1D` | sparse autoencoder (KL penalty) |
| `dae1d` | `DAE1D` | denoising autoencoder |
| `cnn1d` | `CNN1D` | — |
| `lenet1d` | `LeNet1D` | — |
| `resnet18` | `ResNet18` | — |
| `alexnet` | `AlexNet1D` | — |
| `bilstm` | `BiLSTM` | — |

2 protocols x 6 datasets x 9 models = **108 experiments**.

### Fold design

`--folds-source notebook` (default) rebuilds the multiround folds from the round x fold design
printed by the v0 notebooks, stored in `src/fold_designs.json`. It is instant and gives exactly
the folds behind the archived numbers.

`--folds-source generate` runs `FoldIdxGeneratorUnbiased.compute_combinations` for real. It is
correct but materialises `list(combinations(...))` first — for CWRU12k and PU that is
C(256, 4) = 174,792,640 tuples, roughly 15 GB of RAM.

IMS and UOC have no multiround design: their `multi_round_experiments/` notebooks use
`folds_singleround_deep`, so those 18 entries run single round and say so in `notes`.
