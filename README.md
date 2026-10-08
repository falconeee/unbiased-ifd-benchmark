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
run_experiments_ablation.py  entry point: the 63 ablation experiments (see Ablations)

src/                         the benchmark, shared by both entry points
├── registry.py              the 108 experiments (+ experiments.json)
├── datasets.py              the 6 datasets: download, transform, grouping, resampling
├── folds.py                 unbiased folds, single and multi round (+ fold_designs.json)
├── models.py                the 9 architectures
├── experiment.py            DeepLearningExperiment: cross validation and training loop
└── serialize.py             results -> JSON

ablation/                    only what the ablations add on top of src/
├── models.py                parametric 1D-CNN, AlexNet and ResNet-18: depth and first kernel as arguments
├── registry.py              the 63 ablation experiments + their JSON document
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
| `run_experiments_ablation.py` | the 63 ablation experiments | `results_ablation/` |

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
| `--device` | auto-detected | `cuda`, `cuda:N`, `mps` or `cpu` — see [Devices](#devices) |
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

### Devices

Both runners detect the training device on startup and pick the fastest backend torch can
actually use, preferring **CUDA → MPS → CPU**. The choice, and what else was available, is
logged and recorded in every result JSON under `run.environment`:

```
[2026-08-27 22:26:14] device: mps auto-detected -- Apple Silicon GPU (Metal Performance
                      Shaders) (also available: cpu)
```

Override with `--device cuda`, `--device cuda:1`, `--device mps` or `--device cpu`. A request
the machine cannot satisfy fails immediately with a clear message instead of dying inside the
first fold:

```
--device 'cuda': torch reports no usable 'cuda' backend on this machine. Available: mps, cpu.
```

Two things to know about **MPS** (Apple Silicon): results can differ marginally from CPU or
CUDA because the Metal kernels are not bit-identical — this is normal for any accelerator
change and does not affect conclusions; and if an unsupported operator ever aborts a run,
relaunch with `PYTORCH_ENABLE_MPS_FALLBACK=1` set, which routes those ops to the CPU.

One known exception is the benchmark's **`AlexNet1D`**. On torch 2.13, MPS implements
`AdaptiveAvgPool1d` only when the input length is a multiple of the output size
(pytorch/pytorch#96056). AlexNet reaches its `AdaptiveAvgPool1d(6)` with an odd length on
CWRU12k, CWRU48k and PU, so it fails in the first fold, and `PYTORCH_ENABLE_MPS_FALLBACK=1`
does not help. Run those experiments with `--device cpu` or on CUDA. The ablation's AlexNet
variants work around this; see [Ablations](#ablations).

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
time *inside* a model family — the paper's own 1D-CNN, AlexNet and ResNet-18 — and then
checking whether a trend found in one family repeats in the others.

### The grid

Every model runs the same two arms. Variants are named `d<depth>_k<first kernel>`, with depth
counted in the model's own unit; the shared cell (bold) belongs to both arms.

| Model (`--model`) | Depth arm — R2.2 | Receptive-field arm — R2.3 | Shared cell |
|---|---|---|---|
| `cnn1d` | `d1_k3`, `d2_k3`, **`d3_k3`**, `d5_k3` (conv blocks) | **`d3_k3`**, `d3_k7`, `d3_k11`, `d3_k64` | `d3_k3` |
| `alexnet` | `d1_k11`, `d2_k11`, `d3_k11`, **`d5_k11`** (conv layers) | `d5_k3`, `d5_k7`, **`d5_k11`**, `d5_k64` | `d5_k11` = published AlexNet |
| `resnet18` | `d10_k7`, **`d18_k7`**, `d26_k7`, `d34_k7` (weighted layers) | `d18_k3`, **`d18_k7`**, `d18_k11`, `d18_k64` | `d18_k7` = published ResNet-18 |

Because the shared cell is in both arms, each model has 7 distinct variants, not 8. Each
variant is trained once and reported in both tables. 3 models × 7 variants × 3 datasets (PU,
CWRU12k, CWRU48k) = **63 experiments**, each 8 rounds × 4 folds. The same key can name
different networks in different models (`d5_k3` and `d3_k11` exist in both `cnn1d` and
`alexnet`). Result files are prefixed by the model, so they never collide.

What each model holds fixed:

- **`cnn1d`** follows the response letter. Every kernel after the first is 3, channel widths
  are 16→32→64→128→256, and padding, BatchNorm, pooling strategy, dropout and the head are
  identical throughout. `d3_k3` is `src/models.py`'s `CNN1D` with its 7-5-3 kernel schedule
  flattened to 3-3-3. The block structure is otherwise identical, including the
  `AdaptiveMaxPool1d(16)` that replaces the last block's `MaxPool1d(2)`.
- **`alexnet`** is `AlexNet1D` truncated to its first *d* conv layers, with the stem kernel as
  a parameter. Widths are 64-192-384-256-256, later kernels 5-3-3-3, and `MaxPool1d(3, 2)`
  follows conv 1, conv 2 and the last conv. Stem stride 4 and padding 2, no BatchNorm,
  `AdaptiveAvgPool1d(6)` and the 1024-1024 dropout head are as published.
- **`resnet18`** is `ResNet18` with *n* BasicBlocks in each of its four stages (depth =
  2 + 8n: ResNet-10/18/26/34), with the stem kernel as a parameter. Stage widths
  64-128-256-512, the three stride-2 stages, the stem, skip connections, initialisation and
  the head are as published. Blocks are uniform across stages, so "ResNet-34" here is
  `[4, 4, 4, 4]` rather than the canonical `[3, 4, 6, 3]`: same depth, but the arm varies a
  single number.

AlexNet and ResNet-18 are anchored on the published model. Their shared cell *is*
`AlexNet1D` / `ResNet18`, module for module and with the same initial weights under the same
seed, so it can be read against Table 4. The 1D-CNN grid predates that choice and follows the
response letter instead.

| Model | Variant | Depth | First kernel | Params | Receptive field |
|---|---|---|---|---|---|
| `cnn1d` | `d1_k3` | 1 | 3 | 33,508 | 3 |
| | `d2_k3` | 2 | 3 | 67,908 | 8 |
| | `d3_k3` | 3 | 3 | 139,780 | 18 |
| | `d5_k3` | 5 | 3 | 657,028 | 78 |
| | `d3_k7` | 3 | 7 | 139,844 | 22 |
| | `d3_k11` | 3 | 11 | 139,908 | 26 |
| | `d3_k64` | 3 | 64 | 140,756 | 79 |
| `alexnet` | `d1_k11` | 1 | 11 | 1,448,708 | 19 |
| | `d2_k11` | 2 | 11 | 2,296,772 | 67 |
| | `d3_k11` | 3 | 11 | 3,697,988 | 131 |
| | `d5_k11` | 5 | 11 | 3,403,588 | 195 |
| | `d5_k3` | 5 | 3 | 3,403,076 | 187 |
| | `d5_k7` | 5 | 7 | 3,403,332 | 191 |
| | `d5_k64` | 5 | 64 | 3,406,980 | 248 |
| `resnet18` | `d10_k7` | 10 | 7 | 1,753,156 | 195 |
| | `d18_k7` | 18 | 7 | 3,845,956 | 435 |
| | `d26_k7` | 26 | 7 | 5,938,756 | 675 |
| | `d34_k7` | 34 | 7 | 8,031,556 | 915 |
| | `d18_k3` | 18 | 3 | 3,845,700 | 431 |
| | `d18_k11` | 18 | 11 | 3,846,212 | 439 |
| | `d18_k64` | 18 | 64 | 3,849,604 | 492 |

Parameters are for 4 classes. The receptive field is the theoretical one of the convolutional
stack, in input samples. It follows the longest path, so it runs through both convolutions of
every ResNet block, and it excludes the final adaptive pool.

How cleanly each arm isolates its axis, as recorded in every result JSON:

- **The receptive-field arm is close to a clean manipulation in all three models**: parameter
  count moves by less than 1%. The range it covers differs, though. Kernel 3 → 64 multiplies
  the 1D-CNN's receptive field by 4.4×, but AlexNet's only by 1.3× and ResNet-18's by 1.1×,
  because their strided stacks already reach far beyond the first kernel.
- **No depth arm can hold parameters constant**, since adding layers adds weights. ResNet's is
  the cleanest: widths, downsampling and head are fixed, and parameters grow 4.6×. The 1D-CNN's
  grows 20×, because each block also widens the flattened head. AlexNet's is not even
  monotonic (3.70M at depth 3 > 3.40M at depth 5), because the head input is 6 × the last
  conv's width. These confounds are measured (`architecture.num_parameters`) rather than
  hidden. `cnn1d` `d5_k3` vs `d3_k64` is a useful cross-arm reading: near-identical receptive
  field (78 vs 79) with 4.7× the parameters.
- **Initial vs. whole-stack receptive field.** The published ResNet-18 has a larger
  theoretical receptive field (435) than the published AlexNet (195). AlexNet only has the
  larger *first* kernel (11 vs 7). The receptive-field arm tests the first-kernel version of
  the claim.
- AlexNet `d5_k3` uses a stem kernel smaller than the stem stride (3 < 4), so its first layer
  skips one input sample in four. That is part of what the variant measures, and its
  description in the JSON says so.

### Training protocol

Each model trains exactly as its published row does on each dataset. The folds are the same
unbiased multiround folds, the optimizer is Adam, and batch size, learning rate and epochs are
read from that row in `src/experiments.json`: lr 3e-4, batch 64 and 100 epochs everywhere,
except **ResNet-18 on PU, which uses batch 128 and 25 epochs**, as in Table 4. Within an arm
the training configuration is identical, so only the architecture changes. The training loop
is the same `DeepLearningExperiment`. It uses a 20% validation split for logging only and
**no early stopping**. This differs from Section 2.2 of the manuscript, which describes a 10%
split with early stopping. The benchmark code has never done that, and the ablation follows the
code so its numbers stay comparable with Table 4. Both facts are recorded in every result JSON
under `configuration.val_split`, `configuration.early_stopping` and `notes.protocol`.

### Running the AlexNet and ResNet-18 depth arms

The 1D-CNN depth arm already has its PU results (`results_ablation/PU/cnn1d_ablation_d*_k3_pu.json`).
The next step is the depth arm (R2.2) of the two other models. Filters intersect, so the
commands below select exactly those 24 experiments and never touch the 1D-CNN:

```bash
# 1. Check the selection: 24 experiments, AlexNet d1/d2/d3/d5_k11 and ResNet d10/d18/d26/d34_k7
#    on PU, CWRU12k and CWRU48k, with the batch size and epochs each one will use. No cnn1d.
python run_experiments_ablation.py --model alexnet --model resnet18 --study depth --list

# 2. Smoke test, 2 epochs and 1 round, into a separate directory (see the note below).
python run_experiments_ablation.py --model alexnet --study depth --variant d1_k11 \
    --dataset PU --epochs 2 --max-rounds 1 --output-dir /tmp/ablation_smoke

# 3. The whole depth arm of both models, resumable.
python run_experiments_ablation.py --model alexnet --model resnet18 --study depth \
    --resume --keep-going

#    The same, detached, with a transcript.
nohup python run_experiments_ablation.py --model alexnet --model resnet18 --study depth \
    --resume --keep-going > run_ablation_depth.out 2>&1 &

#    Or one model at a time, e.g. AlexNet on the Mac and ResNet-18 on a CUDA machine
#    (see Cost for why), optionally one dataset at a time.
python run_experiments_ablation.py --model alexnet --study depth --resume --keep-going
python run_experiments_ablation.py --model resnet18 --study depth --device cuda --resume --keep-going
python run_experiments_ablation.py --model resnet18 --study depth --dataset PU --device cuda --resume

# 4. Tables and tests for all three models; the 1D-CNN results are picked up automatically.
python run_experiments_ablation.py --report
```

- **`--resume`** skips every variant whose JSON already exists in `--output-dir`, so any of
  these commands can be re-run after a crash or a `Ctrl-C`. To split the work across machines,
  copy each machine's `results_ablation/<DATASET>/*.json` into one `results_ablation/` before
  running `--report`.
- **Keep smoke tests out of `results_ablation/`.** A 2-epoch result written there counts as
  done, and `--resume` would then skip the real run of that variant.
- **The depth arm includes each model's shared cell** (`alexnet d5_k11`, `resnet18 d18_k7`).
  If the receptive-field arm is run later, `--resume` skips that cell instead of training it
  again:
  `python run_experiments_ablation.py --model alexnet --model resnet18 --study kernel --resume --keep-going`.
- **Where to run it.** ResNet-18 needs a CUDA GPU, and on PU one with about 40 GB of memory
  for ResNet-26/34. AlexNet also runs on a Mac but takes about 12 days for the three datasets.
  Details under Cost.

### Cost

**1D-CNN, measured.** The four depth variants on PU took 7.8 h (`d1_k3`), 17.6 h (`d2_k3`),
27.6 h (`d3_k3`) and 61.5 h (`d5_k3`) on an Apple M4 (16 GB, MPS, torch 2.13): 114 h in
total. The estimate this section used to give, about 64 h for all seven PU variants, was low
by more than 2×. The CWRU estimates for the 1D-CNN (≈8.6 h per dataset for all seven variants)
came from the same method, so treat them as optimistic.

**AlexNet and ResNet-18, estimated.** Hours per arm, for all four variants of the arm:

| Arm | Device | PU | CWRU12k | CWRU48k |
|---|---|---|---|---|
| AlexNet depth | CUDA | 4.4 h | 1.6 h | 1.6 h |
| AlexNet depth | MPS (M4) | ≈170 h | ≈60 h | ≈60 h |
| ResNet-18 depth | CUDA | 17 h | 15 h | 16 h |
| ResNet-18 depth | MPS (M4) | does not fit (≈380 h of compute) | ≈330 h | ≈350 h |

The receptive-field arms cost about the same as the shared cell times three more variants.
Beyond the shared cell already trained by the depth arm, AlexNet needs 6.8 / 2.3 / 2.4 h on
CUDA (PU / CWRU12k / CWRU48k) and ResNet-18 needs 10.6 / 9.2 / 9.7 h. Kernel size barely
changes the cost.

Memory at the protocol's batch size (parameters + Adam state + activations) is what decides
where ResNet-18 can run:

| Variant | PU (batch 128) | CWRU48k (batch 64) | CWRU12k (batch 64) |
|---|---|---|---|
| ResNet-10 `d10_k7` | 12.4 GB | 4.7 GB | 1.2 GB |
| ResNet-18 `d18_k7` | 19.4 GB | 7.4 GB | 1.8 GB |
| ResNet-26 `d26_k7` | 26.4 GB | 10.1 GB | 2.5 GB |
| ResNet-34 `d34_k7` | 33.4 GB | 12.8 GB | 3.2 GB |
| any AlexNet | ≤ 2.6 GB | ≤ 2.0 GB | ≤ 0.6 GB |

On a 16 GB Mac, ResNet on PU swaps until it stalls. A ResNet-10 smoke test there filled 20 GB
of swap before finishing its first epoch, and ResNet-26/34 on CWRU48k is borderline.
**Run the ResNet-18 arms on a CUDA GPU.** On PU that means about 40 GB for `d26_k7` and
`d34_k7`, e.g. an A100; a 24 GB card holds `d10_k7` and `d18_k7`. AlexNet fits anywhere. It is
slow on MPS, where PU alone takes about 7 days, but it needs about 8 h on CUDA for all three
datasets.

How these were derived. CUDA hours take the per-epoch times recorded in
`v0_experiments/v0_results/` for the published AlexNet and ResNet-18 (the original Colab GPU
runs) and scale them by each variant's measured cost relative to the published model. MPS hours
on PU are calibrated on the real 27.6 h the 1D-CNN `d3_k3` took on PU on the M4. MPS on CWRU
applies PU's MPS-to-CUDA ratio, so it is the roughest number here. Memory is measured per
sample on MPS and scaled to the batch size; CUDA adds a little for cuDNN workspaces. All
figures assume the datasets are already downloaded and converted.

### Selecting variants

Same idea as the benchmark runner:

| Filter | Accepted values | Notes |
|---|---|---|
| `--model` | `cnn1d`, `alexnet`, `resnet18` | repeatable; without it, all three models |
| `--dataset` | `PU`, `CWRU12k`, `CWRU48k` | repeatable |
| `--study` | `depth`, `kernel` | selects a whole arm; **either arm includes the model's shared cell** |
| `--variant` | `cnn1d`: `d1_k3` `d2_k3` `d3_k3` `d5_k3` `d3_k7` `d3_k11` `d3_k64`<br>`alexnet`: `d1_k11` `d2_k11` `d3_k11` `d5_k11` `d5_k3` `d5_k7` `d5_k64`<br>`resnet18`: `d10_k7` `d18_k7` `d26_k7` `d34_k7` `d18_k3` `d18_k11` `d18_k64` | repeatable; matched in every selected model, so combine with `--model` for `d5_k3` and `d3_k11` |

All filters intersect, so `--model alexnet --study depth --variant d5_k11` runs just the
AlexNet shared cell on each dataset. Passing none runs all 63.

### Commands

```bash
# The whole grid: 63 experiments. Resumable; already-finished variants are skipped.
python run_experiments_ablation.py --all --resume --keep-going

# Detached, with a transcript, for the long run.
nohup python run_experiments_ablation.py --all --resume --keep-going > run_ablation.out 2>&1 &

# Check the selection before spending compute. Prints dataset, model, variant, arm, depth,
# first kernel, batch size and epochs for each experiment, then exits.
python run_experiments_ablation.py --all --list

# One model, one dataset (7 experiments).
python run_experiments_ablation.py --model cnn1d --dataset PU --resume

# One arm of one model on one dataset (4 experiments, including the shared cell).
python run_experiments_ablation.py --model cnn1d --dataset PU --study depth     # R2.2
python run_experiments_ablation.py --model cnn1d --dataset PU --study kernel    # R2.3

# The depth arm of AlexNet and ResNet-18 -- see "Running the AlexNet and ResNet-18 depth arms".
python run_experiments_ablation.py --model alexnet --model resnet18 --study depth --resume --keep-going

# The receptive-field arm of AlexNet and ResNet-18.
python run_experiments_ablation.py --model alexnet --model resnet18 --study kernel --resume --keep-going

# Smoke test: one variant, 2 epochs, 1 round, in a scratch directory so --resume never
# mistakes it for a finished result. The overrides are recorded in run.overrides.
python run_experiments_ablation.py --model cnn1d --dataset CWRU12k --variant d1_k3 \
    --epochs 2 --max-rounds 1 --output-dir /tmp/ablation_smoke

# Rebuild report.md / report.tex / report.json from results already on disk,
# without training anything. Safe to run at any time, including mid-run.
python run_experiments_ablation.py --report
```

### Flags that differ

Everything from the benchmark runner's [Flags](#flags) applies unchanged — `--data-root`,
`--output-dir`, `--artifacts-dir`, `--log-file`, `--no-download`, `--device`, `--seed`,
`--resume`, `--keep-going`, `--no-artifacts`, `--folds-source`, `--epochs`, `--batch-size`,
`--max-rounds`. The differences:

| Flag | Default | Notes |
|---|---|---|
| `--model` | all three | selects a model *family* (`cnn1d`, `alexnet`, `resnet18`), not one network |
| `--variant`, `--study` | — | replace the benchmark's per-model selection; see above |
| `--output-dir` | `results_ablation/` | not `results/`; keep them apart |
| `--epochs`, `--batch-size`, `--max-rounds` | — | the only overrides; there is no `--pretrain-epochs` because no ablation variant pre-trains |
| `--report` | off | rebuild the report from `--output-dir` and exit without training |

Device detection is identical to the benchmark runner — CUDA, then MPS, then CPU; see
[Devices](#devices). The AlexNet variants run on MPS even though the benchmark's `AlexNet1D`
does not (see Devices). Their adaptive average pool switches to an equivalent matmul only on
MPS and only when the input length is not a multiple of 6. Everywhere else it is the native
op.

`--data-root` defaults to the same `data/` as the benchmark, so nothing is downloaded twice.

### Output

`results_ablation/` mirrors `results/` one level flatter (no suite directory), with an extra
top-level `architecture` block per experiment and `family`/`study`/`variant` fields:

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
    └── <model>_ablation_<variant>_<dataset>.json   e.g. alexnet_ablation_d3_k11_pu.json
```

Each experiment JSON is the benchmark schema plus:

| Key | Contents |
|---|---|
| `family`, `study`, `variant` | the model family (`cnn1d`, `alexnet`, `resnet18`), which arm the variant belongs to (`depth`, `kernel` or `both`) and its name. 1D-CNN results written before `family` existed carry it in `suite` (`cnn1d_ablation`), which the report understands |
| `source` | the review comment it answers (`R2.2`, `R2.3`) and the published experiment to compare against |
| `architecture` | depth, first kernel, parameter counts (total, conv, head), theoretical receptive field in samples and seconds, plus per model: kernels and channels per block (`cnn1d`, `alexnet`), stem stride/padding and max-pool count (`alexnet`), blocks per stage and stage widths (`resnet18`) |
| `configuration` | adds `optimizer`, `loss`, `val_split`, `early_stopping`, `checkpoint_selection` |
| `notes` | why the protocol and the model's structure are what they are |

### Reading the report

The report is regenerated at the end of every run, and by `--report` alone. It is organised
dataset → model → arm, and each dataset opens with a **cross-model summary**: one table per arm,
one row per model, with balanced accuracy at each level, the best variant and the trend
statistics. It is the table that answers whether a trend repeats across architectures. Each row
is a within-model test; nothing is tested *across* models, because their variants are different
networks. Per model and per arm it then gives:

- **Mean ± std** of Balanced Accuracy and Macro F1, under *both* dispersion conventions — std
  over all 32 per-fold scores (what Table 4's caption claims) and std over the 8 round means
  (what `results.summary` stores) — because the two differ and the manuscript is ambiguous.
- **Adjacent paired Wilcoxon** tests between neighbouring variants, paired by `(round, fold)`
  and Holm-corrected within the arm. Pairing is legitimate because every variant sees the
  identical fold assignment with the identical seed.
- **Page's trend test** across the whole arm, run in both directions — this is the test that
  actually addresses "monotonic in depth", which adjacent pairwise tests do not.
- **Spearman ρ** against depth or kernel size, for effect direction and magnitude.

An arm with fewer than two finished variants is left out, and a partial arm is flagged with the
variants it is missing, so the report can be built at any point of a run.

`report.tex` contains the same tables as `booktabs` environments, ready to paste into the
manuscript: `\label{tab:ablation_summary_<arm>_<dataset>}` for the cross-model summaries and
`\label{tab:ablation_<model>_<arm>_<dataset>}` for the per-model tables.

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
