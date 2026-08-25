# unbiased-ifd-benchmark

A benchmark for **intelligent fault diagnosis** (IFD) on rolling-bearing vibration signals,
run under an **unbiased** evaluation protocol: instead of splitting samples at random, each
test fold holds out whole acquisition groups — motor load for CWRU, operating condition for
PU, bearing and test rig for IMS, fault severity for UOC. A model therefore has to generalise
to a condition it never saw, rather than to another window of a recording it already
memorised.

Nine 1D deep-learning architectures across six public datasets, in two protocols —
**108 experiments** in total.

```bash
uv venv --no-project
uv pip install -r requirements.txt

nohup uv run --no-project python run_experiments.py --all --resume --keep-going > run.out 2>&1 &
```

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

---

## The benchmark

Rolling-bearing **intelligent fault diagnosis**: classify a vibration signal by the fault it
comes from. The point of this repository is the *evaluation protocol*, not the models.

**Why "unbiased".** Splitting vibration windows at random leaks information: consecutive
windows of one recording end up on both sides of the split, and a model can score near 100%
by recognising the recording rather than the fault. Here every test fold holds out whole
acquisition groups instead:

| Dataset | Grouped by | Folds |
|---|---|---|
| CWRU 12k / 48k | motor load (0–3 hp) | 4 |
| IMS | test rig and bearing | 3 |
| MFPT | fault type and load, binned | 3 (single round) / 7 (multi round) |
| PU | rotation speed, load torque, radial force | 4 |
| UOC | fault severity | 5 |

The scores this produces are far below what these datasets usually report. In the v0 results,
IMS lands near chance for every architecture (0.14–0.24 accuracy over 7 classes) and the
MLP/autoencoder family sits around 0.44 on CWRU12k over 4 classes, while the convolutional
models hold up better (CNN1D 0.88, ResNet18 0.93). Measuring that spread is the point.

**Two protocols.**

- **single round** — one unbiased fold assignment, one cross validation.
- **multi round** — the condition-to-fold mapping is rotated several times (8 rounds for
  CWRU12k/CWRU48k/PU, 5 for MFPT) and the whole cross validation is repeated per round, so
  the score does not depend on one lucky arrangement of conditions.

**Nine architectures**, four of them with an unsupervised pre-training phase:

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

**Where this came from.** The benchmark started as twelve Colab notebooks, archived unchanged
in `v0_experiments/`. `run_experiments.py` is their script form: same datasets, same folds,
same architectures, same hyperparameters (which are *not* uniform — `pretrain_epochs` is 50
for some datasets and 100 for others, and `multi_round/PU` trains ResNet18 with
`batch_size=128` for 25 epochs). `v0_experiments/v0_results/` holds the results parsed out of
the notebook cell outputs, in the same JSON schema this script writes, so old and new runs can
be compared field by field. Eleven bugs found in the notebook code are listed at the bottom of
this file.

---

## Install

[uv](https://docs.astral.sh/uv/) is the recommended way — resolving and installing torch and
the rest takes seconds instead of minutes. From the repository root:

```bash
uv venv --no-project                # creates ./.venv
uv pip install -r requirements.txt  # installs into ./.venv
```

The repository has no `pyproject.toml`, hence `--no-project`; `uv pip install` picks up
`./.venv` on its own, no activation needed.

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

`uv run --no-project` uses `./.venv` without activating anything, which is what you want for
a run you leave going in the background:

```bash
# everything (108 experiments), resumable, in the background
nohup uv run --no-project python run_experiments.py --all --resume --keep-going > run.out 2>&1 &

# one dataset, one protocol
uv run --no-project python run_experiments.py --suite single_round --dataset CWRU12k

# one experiment, smoke test
uv run --no-project python run_experiments.py --suite single_round --dataset UOC \
    --model cnn1d --epochs 2 --max-rounds 1

# what would run, without running it
uv run --no-project python run_experiments.py --all --list
```

Equivalent without uv — activate the environment first (`source .venv/bin/activate`) and drop
the `uv run --no-project` prefix, or call the interpreter directly:

```bash
nohup .venv/bin/python run_experiments.py --all --resume --keep-going > run.out 2>&1 &
```

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
| `--folds-source` | `notebook` | see [Multiround folds](#multiround-folds) |
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
| `results.fold_errors` | present only when a fold failed (see fix E) |

Multiround experiments replace `folds_training_log` with `rounds[]`, each round carrying its
own summary, per-fold metrics, confusion matrix and training logs, plus a `fold_design`
describing the round x fold group assignment.

`v0_experiments/v0_results/` uses the same schema, so a new run can be diffed against the
notebook numbers directly. The v0 files have no `run` block and their training history only
has every fifth epoch — that is all the notebooks printed.

## Multiround folds

`--folds-source notebook` (default) rebuilds the multiround folds from the round x fold
design the v0 notebooks printed, stored in `src/fold_designs.json`. It is instant and gives
exactly the folds behind the v0 numbers.

`--folds-source generate` runs `FoldIdxGeneratorUnbiased.compute_combinations` for real. It is
correct but materialises `list(combinations(...))` first — for CWRU12k and PU that is
C(256, 4) = 174,792,640 tuples, roughly 15 GB of RAM.

IMS and UOC have no multiround design: their `multi_round_experiments/` notebooks use
`folds_singleround_deep`, so those 18 entries run single round and say so in `notes`.

---

## Fixes applied to the notebook code

The `DeepLearningExperiment` class lives inside the notebook cells (it is not in
`signalAI==0.0.8`, and the version on SignAI-Framework `main` has different semantics — no
autoencoder pre-training phase). It was copied to `src/experiment.py`; every deviation is
marked `# FIX:` in the source. The training maths is unchanged, so results stay comparable —
epoch-1 pre-training losses reproduce the notebooks' (script vs notebook: ae1d 0.1191 vs
0.1188, sae1d 29.37 vs 29.43, dae1d 0.1229 vs 0.1235; the gap is shuffling noise).

**A. `FoldIdxGeneratorUnbiased.generate_folds()` crashes in multiround mode.**
With `multiround=True` the trailing validation runs `max(folds)` over a list of numpy
arrays: `ValueError: The truth value of an array with more than one element is ambiguous`.
Only `multi_round/CWRU12k` calls it that way (the other notebooks call
`generate_folds_unbiased_multiround()`, which skips the validation), which suggests that cell
was edited after it was last executed. The runner always calls the multiround method
directly.

**B. Multiround fold generation needs ~15 GB of RAM.** See [Multiround folds](#multiround-folds).

**C. Every notebook converts into the same directory.** All twelve use
`deep_root_dir_time = "../data/deep_data/deep_learning"`. `convertDataset` hashes only the
raw class, its length and the transforms — **not the `filter`** — so running more than one
dataset against one directory either raises `ValueError: Dataset corrupted!` or silently
returns another dataset's cache. CWRU12k and CWRU48k differ *only* by the filter, so they
would collide outright. Each dataset now converts into `<data-root>/deep_data/<DATASET>/`.

**D. `prepare_data()` runs inside `__init__`.** In the multiround loop a fresh
`DeepLearningExperiment` is built per round, so the whole `DeepDataset` is read from disk
once per round — 8x per model, 72x per dataset. The runner loads `X`/`y` once per dataset
and injects them, and casts `X` to float32 (the notebook's `np.array(features)` stays
float64: 242 MB for CWRU12k alone).

**E. A failed fold disappears from the averages.** `run()` catches `Exception`, prints it and
continues; the fold never reaches `results.folds` and the mean is computed over the
survivors, with nothing in the saved results to say so. The resilience is kept, but the
failure is now recorded in `results.fold_errors`.

**F. A trailing batch of one sample kills the fold.** Without `drop_last`, when
`len(train) % batch_size == 1` `nn.BatchNorm1d` raises *"Expected more than 1 value per
channel when training"* — landing straight in the handler from fix E. `drop_last=True` is
now set for that case only, so nothing else changes.

**G. The `nn.DataParallel` branch is dead code.** `super().__init__(..., model=model)` stores
`self.model` *before* `model = nn.DataParallel(model)` rebinds the local, and
`_train_one_fold` deep-copies `self.model` — the wrapper never reached training. Removed;
the device is chosen explicitly with `--device`.

**H. The validation split was not reproducible.** `random_split` draws from torch's global
RNG, which the notebooks never seed, so every run got a different split. `--seed` (42 by
default) seeds python/numpy/torch, and each fold gets its own `torch.Generator`.

**I. `val_size` can be 0.** `int(0.2 * len(train))` is 0 for small folds, and the epoch loop
then divides by `len(val_loader.dataset)`. It is clamped to at least one sample, with an
explicit error if the fold is too small to split at all.

**J. Matplotlib.** `plt.figure()` per fold with no `plt.close()` leaks a figure per fold, and
`show_results()` calls `plt.show()`, which needs an interactive backend. The runner forces
`Agg` and closes each figure.

**K. `self.model.to(self.device)` mutates the shared prototype** and leaves an extra copy on
the GPU. The prototype now stays on CPU; only each fold's copy moves to the device.

### Left as-is (behaviour, not bugs)

- Per-fold `accuracy` is `balanced_accuracy_score`, while the pooled "overall accuracy" of the
  confusion matrix is the raw one — that is why the two numbers differ.
- There is no early stopping or checkpoint selection: the model tested is the one from the
  last epoch, even though validation loss diverges well before that.
- The validation split ignores groups, so it is not unbiased — only the outer test folds are.
- `recon_loss_weight` is accepted and stored but never read: the pre-training loss is the
  reconstruction criterion on its own. The notebooks pass `1.0`, so nothing changes either
  way, but the knob does not do anything.
