# unbiased-ifd-benchmark

Data and code behind the paper: a benchmark for **intelligent fault diagnosis** (IFD) on
rolling-bearing vibration signals, evaluated under an **unbiased** protocol. Instead of
splitting samples at random, every test fold holds out whole acquisition groups — motor load
for CWRU, operating condition for PU, bearing and test rig for IMS, fault severity for UOC. A
model therefore has to generalise to a condition it never saw, rather than to another window of
a recording it already memorised.

Nine 1D deep-learning architectures across six public datasets, in two protocols —
**108 experiments** in total.

```bash
uv venv --no-project
source .venv/bin/activate
uv pip install -r requirements.txt
python run_experiments.py --all --resume --keep-going
```

---

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
run_experiments.py     the entry point: selection, orchestration, logging, result files
src/                   everything it calls
├── registry.py        the 108 experiments (+ experiments.json)
├── datasets.py        the 6 datasets: download, transform, grouping, resampling
├── folds.py           unbiased folds, single and multi round (+ fold_designs.json)
├── models.py          the 9 architectures
├── experiment.py      DeepLearningExperiment: cross validation and training loop
└── serialize.py       results -> JSON
results/               output of a run (created on first run)
data/                  raw downloads and converted datasets (git-ignored)
v0_experiments/        archived first version: the original notebooks and their results
requirements.txt
```

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

With the environment activated:

```bash
# everything (108 experiments), resumable
python run_experiments.py --all --resume --keep-going

# leave it going in the background
nohup python run_experiments.py --all --resume --keep-going > run.out 2>&1 &

# one dataset, one protocol
python run_experiments.py --suite single_round --dataset CWRU12k

# one experiment, smoke test
python run_experiments.py --suite single_round --dataset UOC \
    --model cnn1d --epochs 2 --max-rounds 1

# what would run, without running it
python run_experiments.py --all --list
```

`uv run --no-project python run_experiments.py ...` works too, and uses `./.venv` without
activating anything.

`--suite`, `--dataset` and `--model` are repeatable and combine as a filter. Raw datasets are
downloaded on first use into `data/` and converted once per dataset, so the first run of a
given dataset is slower than the rest.

### Flags

| Flag | Default | Notes |
|---|---|---|
| `--suite`, `--dataset`, `--model` | all | repeatable selection filters |
| `--list` | — | print the selection and exit |
| `--data-root` | `data/` | raw downloads, converted datasets, group/fold caches |
| `--output-dir` | `results/` | result JSONs, `run.log`, `run_manifest.json`, `_artifacts/` |
| `--resume` | off | skip experiments whose JSON already exists |
| `--keep-going` | off | carry on when one experiment raises |
| `--device` | `cuda` if available | |
| `--seed` | `42` | seeds python/numpy/torch and the validation split |
| `--folds-source` | `notebook` | see [Fold design](#fold-design) |
| `--no-artifacts` | off | skip per-fold loss curves and `.pt` checkpoints |
| `--no-download` | off | fail instead of downloading a missing dataset |
| `--epochs`, `--pretrain-epochs`, `--batch-size`, `--max-rounds` | — | overrides for smoke tests; recorded in the JSON under `run.overrides` |

Progress goes to stdout **and** to `<output-dir>/run.log`, so a backgrounded run keeps a full
transcript. Re-running with `--resume` picks up where it stopped — including after a crash or
a `Ctrl-C`, since results are written per experiment, not at the end.

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
