# scripts/

Two entry points:

| Script | What it does |
|---|---|
| `run_experiments.py` | Runs the benchmark experiments and writes one JSON per experiment into `results/`. Script form of the notebooks — meant to be left running in the background. |
| `extract_experiment_outputs.py` | Parses the outputs already stored in the notebooks and writes them to `output/`. Read-only over the `.ipynb` files. |

---

## Install

[uv](https://docs.astral.sh/uv/) is the recommended way — resolving and installing torch and
the rest takes seconds instead of minutes. From the repository root:

```bash
uv venv --no-project                        # creates ./.venv
uv pip install -r scripts/requirements.txt  # installs into ./.venv
```

The repository has no `pyproject.toml`, hence `--no-project`; `uv pip install` picks up
`./.venv` on its own, no activation needed.

<details>
<summary>Without uv</summary>

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r scripts/requirements.txt
```
</details>

`gdown<5` is not optional: `vibdata==1.1.1` calls `gdown.cached_download(..., md5=...)`, and
gdown 5 removed that argument, so every raw download fails with
`TypeError: download() got an unexpected keyword argument 'md5'`.

## Run

`uv run --no-project` uses `./.venv` without activating anything, which is what you want for
a run you leave going in the background:

```bash
# everything (108 experiments), resumable, in the background
nohup uv run --no-project python scripts/run_experiments.py \
    --all --resume --keep-going > run.out 2>&1 &

# one notebook
uv run --no-project python scripts/run_experiments.py --suite single_round --dataset CWRU12k

# one experiment, smoke test
uv run --no-project python scripts/run_experiments.py --suite single_round --dataset UOC \
    --model cnn1d --epochs 2 --max-rounds 1

# what would run, without running it
uv run --no-project python scripts/run_experiments.py --all --list
```

Equivalent without uv — activate the environment first (`source .venv/bin/activate`) and drop
the `uv run --no-project` prefix, or call the interpreter directly:

```bash
nohup .venv/bin/python scripts/run_experiments.py --all --resume --keep-going > run.out 2>&1 &
```

`--suite`, `--dataset` and `--model` are repeatable and combine as a filter. The full
selection is 2 suites x 6 datasets x 9 models = 108 experiments.

| Model key | Class | Pre-training |
|---|---|---|
| `mlp1d` | `MLP1D` | — |
| `ae1d` | `AE1D` | autoencoder |
| `sae1d` | `SAE1D` | sparse autoencoder (KL) |
| `dae1d` | `DAE1D` | denoising autoencoder |
| `cnn1d` | `CNN1D` | — |
| `lenet1d` | `LeNet1D` | — |
| `resnet18` | `ResNet18` | — |
| `alexnet` | `AlexNet1D` | — |
| `bilstm` | `BiLSTM` | — |

### Useful flags

| Flag | Default | Notes |
|---|---|---|
| `--data-root` | `data/` | raw downloads, converted datasets, group/fold caches |
| `--output-dir` | `results/` | result JSONs, `run.log`, `run_manifest.json`, `_artifacts/` |
| `--resume` | off | skips experiments whose JSON already exists |
| `--keep-going` | off | carries on when one experiment raises |
| `--device` | `cuda` if available | |
| `--seed` | `42` | seeds python/numpy/torch and the validation split |
| `--folds-source` | `notebook` | see "Multiround folds" below |
| `--no-artifacts` | off | skip per-fold loss curves and `.pt` checkpoints |
| `--epochs`, `--pretrain-epochs`, `--batch-size`, `--max-rounds` | — | overrides, for smoke tests; recorded in the JSON under `run.overrides` |

Progress goes to stdout **and** to `<output-dir>/run.log`, so a backgrounded run keeps a full
transcript. Re-running with `--resume` picks up where it stopped.

## Output

Same layout and schema as `output/` (documented in `output/README.md`):

```
results/
├── index.json
├── run.log
├── run_manifest.json
├── _artifacts/<suite>/<DATASET>/<experiment>/   # loss curves + model checkpoints
└── <suite>/<DATASET>/
    ├── _index.json
    └── <experiment_name>.json
```

Each experiment JSON carries everything the notebook-derived ones do, plus what only a live
run can produce:

- `run` — timestamps, duration, seed, device, fold source, overrides, package versions
- `results.*.training_history` — **every** epoch (the notebooks only printed every 5th)
- per-fold `confusion_matrix`, `precision`, `recall`, `roc_auc`
- `results.*.duration_seconds` per experiment and per round
- `results.*.fold_errors` when a fold fails (see fix E)

## Multiround folds

`--folds-source notebook` (default) rebuilds the multiround folds from the round x fold
design the notebooks printed, stored in `runner/fold_designs.json`. It is instant and gives
exactly the folds behind the published numbers.

`--folds-source generate` runs `FoldIdxGeneratorUnbiased.compute_combinations` for real. It
is correct but materialises `list(combinations(...))` first — for CWRU12k and PU that is
C(256, 4) = 174,792,640 tuples, roughly 15 GB of RAM.

IMS and UOC have no multiround design: their `multi_round_experiments/` notebooks use
`folds_singleround_deep`, so those 18 entries run single round and say so in `notes`.

---

## Fixes applied to the notebook code

The `DeepLearningExperiment` class lives inside the notebook cells (it is not in
`signalAI==0.0.8`, and the version on SignAI-Framework `main` has different semantics — no
autoencoder pre-training phase). It was copied to `runner/experiment.py`; every deviation is
marked `# FIX:` in the source. The training maths is unchanged, so results stay comparable.

**A. `FoldIdxGeneratorUnbiased.generate_folds()` crashes in multiround mode.**
With `multiround=True` the trailing validation runs `max(folds)` over a list of numpy
arrays: `ValueError: The truth value of an array with more than one element is ambiguous`.
Only `multi_round/CWRU12k` calls it that way (the other notebooks call
`generate_folds_unbiased_multiround()`, which skips the validation), which suggests that cell
was edited after it was last executed. The runner always calls the multiround method
directly.

**B. Multiround fold generation needs ~15 GB of RAM.** See "Multiround folds" above.

**C. Every notebook converts into the same directory.** All twelve use
`deep_root_dir_time = "../data/deep_data/deep_learning"`. `convertDataset` hashes only the
raw class, its length and the transforms — **not the `filter`** — so running more than one
dataset against one directory either raises `ValueError: Dataset corrupted!` or silently
returns another dataset's cache. CWRU12k and CWRU48k differ *only* by the filter, so they
would collide outright. Each dataset now converts into `<data-root>/deep_data/<DATASET>/`.

**D. `prepare_data()` runs inside `__init__`.** In the multiround loop a fresh
`DeepLearningExperiment` is built per round, so the whole `DeepDataset` is read from disk
once per round — 8x per model, 72x per dataset. The runner loads `X`/`y` once per notebook
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

- Per-fold `accuracy` is `balanced_accuracy_score`, while `show_results()`'s "Acurácia Geral"
  is the raw accuracy of the pooled confusion matrix — that is why the two numbers differ.
- There is no early stopping or checkpoint selection: the model tested is the one from the
  last epoch, even though validation loss diverges well before that.
- The validation split ignores groups, so it is not unbiased — only the outer test folds are.
- `recon_loss_weight` is accepted and stored but never read: the pre-training loss is the
  reconstruction criterion on its own. The notebooks pass `1.0`, so nothing changes either
  way, but the knob does not do anything.
