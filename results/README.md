# results/

Every result reported in the paper, as produced by the experiments, plus tables and figures
derived from them.

| Folder | Contents |
|---|---|
| [`benchmark/`](benchmark/) | the 108 benchmark experiments (2 protocols × 6 datasets × 9 models), extracted from the cell outputs of the original Colab notebooks in `v0_experiments/`; see [`benchmark/README.md`](benchmark/README.md) |
| [`ablation/`](ablation/) | the 63 ablation experiments written by `run_experiments_ablation.py`, plus `report.md`, `report.tex` and `report.json` (tables and significance tests) |
| [`tables/`](tables/) | Tables 4, 5 and 6 of the paper as CSV, each row pointing to its source JSON |
| [`figures/`](figures/) | the ablation figures of the paper (Figures 1–3) and the benchmark figure of the main README, as PDF and PNG |

## Paper → files

| In the paper | Source files | Derived |
|---|---|---|
| Table 4: benchmark, 9 models × 4 datasets | `benchmark/multi_round/{CWRU12k,CWRU48k,MFPT,PU}/*.json` | `tables/table4_benchmark.csv`, `figures/benchmark.*` |
| Table 5: ablation, 21 variants × 3 datasets | `ablation/{PU,CWRU12k,CWRU48k}/*_ablation_*.json` | `tables/table5_ablation.csv` |
| Table 6: trend and adjacent-step tests | same as Table 5 | `tables/table6_ablation_tests.csv`, `ablation/report.md` |
| Figure 1: depth arm | same as Table 5 | `figures/ablation_depth.*` |
| Figure 2: kernel arm | same as Table 5 | `figures/ablation_kernel.*` |
| Figure 3: all variants per family, with the base models of Table 4 | Table 5 files + Table 4 files of 1D-CNN, AlexNet and ResNet-18 | `figures/ablation_ranges.*` |

Benchmark file names come from the notebooks and are not always the dataset's name. For
example, CWRU-48k uses `cnn1d_vibration.json` and `alexnet_12k.json`. The `source` column of
`tables/table4_benchmark.csv` gives the exact file for every row.

## How the numbers are computed

- **Metrics.** Per-fold `accuracy` is scikit-learn's `balanced_accuracy_score` (Balanced
  Accuracy in the paper), and `f1_score` is the macro F1. Both are stored as fractions; the
  tables use %.
- **Mean ± std.** The mean is taken over all folds of all rounds (8 rounds × 4 folds; MFPT
  5 rounds × 7 folds). The ± is the population standard deviation (ddof = 0) of the per-round
  means. This is what `results.summary` stores in each JSON and what the paper reports.
  `ablation/report.md` also shows the standard deviation over all 32 folds.
- **Tests (Table 6).** Within each family and arm, the four levels are compared on the 32
  paired folds (same folds and seed for every variant). Page's trend test is run in both
  directions; the CSV gives the direction that is significant at α = 0.05 (or `none`) and the
  smaller one-sided p-value. Adjacent levels are compared with the two-sided Wilcoxon
  signed-rank test, Holm-corrected within the arm. Differences are in percentage points.
- **Rounding.** Values are rounded to two decimals, as in the paper.

## Notes on the archived results

- The single-round PU notebook has no stored output, so its nine experiments are
  `status: not_executed`. In the multi-round PU notebook, MLP1D, AE1D, SAE1D and DAE1D were not
  run, because those models had collapsed on the other datasets. These are the dashes of
  Table 4.
- IMS and UOC have no multi-round fold design. Their multi-round notebooks run single round,
  and the JSONs say so in `notes` and `protocol`. Neither dataset is in the paper.
- The 63 ablation JSONs carry `source.review_file: "review/reviwers_comments.md"`, the file the
  experiments were designed from. That reviewer report is not distributed with this repository.
- The archived benchmark JSONs keep only every fifth epoch of the training history, because
  that is all the notebooks printed. The ablation JSONs keep every epoch.

## All six datasets

The paper reports four datasets under the multi-round protocol. The archive also covers IMS
and UOC, and a single-round protocol. Mean Balanced Accuracy over the unbiased folds, as
fractions; the class count is in the header because it sets the chance level.

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

IMS and UOC have no multi-round design, so their multi-round entries repeat the single-round
result.

- **IMS is at chance for every architecture** (0.14–0.24 over 7 classes). Holding out the test
  rig and bearing leaves nothing transferable in this feature space.
- **UOC stays easy for every architecture** (0.80–0.98), even with the fully connected and
  autoencoder models that collapse elsewhere.
- **Multi round shifts the numbers by a few points but not the ranking**, mostly downward for
  the weaker models: the single-round figure tends to be the optimistic one.
