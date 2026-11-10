# Ablations -- depth (R2.2) and receptive field (R2.3)

Values are percentages, `mean ± std` over **all per-fold scores** (the convention Table 4's caption states). `std over rounds` is the alternative convention stored in `results.summary` of each JSON, shown for transparency.

`Balanced Accuracy` is `balanced_accuracy_score`; `Macro F1` is `f1_score(average='macro')`.

Depth is counted in each model's own unit: conv blocks for the 1D-CNN, conv layers for AlexNet, weighted layers for ResNet (10/18/26/34).

## PU

### Cross-model summary -- R2.2 -- depth ablation

| Model | Axis | B.Acc by level | Best | ρ B.Acc | Page p ↑ / ↓ (B.Acc) | ρ F1 | Page p ↑ / ↓ (F1) |
|---|---|---|---|---|---|---|---|
| 1D-CNN | conv blocks | 1: 61.23 · 2: 66.21 · 3: 68.85 · 5: 72.99 | `d5_k3` | +0.292 | <0.001 / 1.000 | +0.208 | <0.001 / 1.000 |
| AlexNet | conv layers | 1: 69.16 · 2: 70.47 · 3: 70.64 · 5: 71.44 | `d5_k11` | +0.038 | 0.049 / 0.951 | +0.035 | 0.043 / 0.957 |
| ResNet-18 | weighted layers | 10: 61.38 · 18: 63.21 · 26: 64.20 · 34: 64.62 | `d34_k7` | +0.055 | <0.001 / 1.000 | +0.058 | 0.004 / 0.996 |

### Cross-model summary -- R2.3 -- receptive field ablation

| Model | Axis | B.Acc by level | Best | ρ B.Acc | Page p ↑ / ↓ (B.Acc) | ρ F1 | Page p ↑ / ↓ (F1) |
|---|---|---|---|---|---|---|---|
| 1D-CNN | first kernel size | 3: 68.85 · 7: 70.76 · 11: 69.24 · 64: 68.32 | `d3_k7` | -0.024 | 0.750 / 0.250 | -0.036 | 0.769 / 0.231 |
| AlexNet | first kernel size | 3: 74.25 · 7: 71.35 · 11: 71.44 · 64: 69.05 | `d5_k3` | -0.095 | 0.994 / 0.006 | -0.098 | 0.998 / 0.002 |
| ResNet-18 | first kernel size | 3: 63.12 · 7: 63.21 · 11: 63.90 · 64: 63.90 | `d18_k11` | +0.014 | 0.250 / 0.750 | +0.036 | 0.099 / 0.901 |

B.Acc by level is the mean over all per-fold scores, in %. ρ is Spearman's against the arm's level; Page p ↑ tests a trend increasing with the level, ↓ a decreasing one. Each row is a separate within-model test.

### 1D-CNN

#### R2.2 -- depth ablation

| Variant | Depth | Params | RF (samples) | B.Acc | Macro F1 | B.Acc (std over rounds) |
|---|---|---|---|---|---|---|
| `d1_k3` | 1 | 33,508 | 3 | 61.23 ± 13.76 | 60.07 ± 18.58 | 61.23 ± 1.35 |
| `d2_k3` | 2 | 67,908 | 8 | 66.21 ± 16.19 | 64.26 ± 20.67 | 66.21 ± 2.59 |
| `d3_k3` | 3 | 139,780 | 18 | 68.85 ± 18.13 | 65.49 ± 21.94 | 68.85 ± 2.55 |
| `d5_k3` | 5 | 657,028 | 78 | 72.99 ± 16.45 | 70.26 ± 18.96 | 72.99 ± 3.37 |

**Balanced Accuracy -- adjacent paired Wilcoxon** (n pairs = folds shared by both variants; Holm-corrected within this arm)

| Comparison | Δ mean (pp) | Δ median (pp) | n | p | p (Holm) | sig. |
|---|---|---|---|---|---|---|
| `d1_k3` → `d2_k3` | +4.98 | +4.86 | 32 | <0.001 | <0.001 | yes |
| `d2_k3` → `d3_k3` | +2.64 | +1.90 | 32 | 0.017 | 0.017 | yes |
| `d3_k3` → `d5_k3` | +4.14 | +3.19 | 32 | <0.001 | <0.001 | yes |

**Balanced Accuracy -- trend across conv blocks** (Page's L over 32 folds x 4 variants)

- Page's L, increasing with conv blocks: L = 905.0, p = <0.001
- Page's L, decreasing with conv blocks: L = 695.0, p = 1.000
- Spearman ρ vs conv blocks: +0.292, p = <0.001, n = 128

**Macro F1 -- adjacent paired Wilcoxon** (n pairs = folds shared by both variants; Holm-corrected within this arm)

| Comparison | Δ mean (pp) | Δ median (pp) | n | p | p (Holm) | sig. |
|---|---|---|---|---|---|---|
| `d1_k3` → `d2_k3` | +4.19 | +4.83 | 32 | <0.001 | 0.002 | yes |
| `d2_k3` → `d3_k3` | +1.23 | +0.24 | 32 | 0.340 | 0.340 | no |
| `d3_k3` → `d5_k3` | +4.77 | +3.66 | 32 | <0.001 | <0.001 | yes |

**Macro F1 -- trend across conv blocks** (Page's L over 32 folds x 4 variants)

- Page's L, increasing with conv blocks: L = 884.0, p = <0.001
- Page's L, decreasing with conv blocks: L = 716.0, p = 1.000
- Spearman ρ vs conv blocks: +0.208, p = 0.018, n = 128

#### R2.3 -- receptive field ablation

| Variant | First kernel | Params | RF (samples) | B.Acc | Macro F1 | B.Acc (std over rounds) |
|---|---|---|---|---|---|---|
| `d3_k3` | 3 | 139,780 | 18 | 68.85 ± 18.13 | 65.49 ± 21.94 | 68.85 ± 2.55 |
| `d3_k7` | 7 | 139,844 | 22 | 70.76 ± 16.76 | 67.64 ± 20.30 | 70.76 ± 3.87 |
| `d3_k11` | 11 | 139,908 | 26 | 69.24 ± 17.33 | 66.08 ± 20.48 | 69.24 ± 2.81 |
| `d3_k64` | 64 | 140,756 | 79 | 68.32 ± 18.43 | 64.16 ± 21.17 | 68.32 ± 3.57 |

**Balanced Accuracy -- adjacent paired Wilcoxon** (n pairs = folds shared by both variants; Holm-corrected within this arm)

| Comparison | Δ mean (pp) | Δ median (pp) | n | p | p (Holm) | sig. |
|---|---|---|---|---|---|---|
| `d3_k3` → `d3_k7` | +1.91 | +0.71 | 32 | 0.048 | 0.143 | no |
| `d3_k7` → `d3_k11` | -1.52 | +0.04 | 32 | 0.421 | 0.700 | no |
| `d3_k11` → `d3_k64` | -0.92 | -0.45 | 32 | 0.350 | 0.700 | no |

**Balanced Accuracy -- trend across first kernel size** (Page's L over 32 folds x 4 variants)

- Page's L, increasing with first kernel size: L = 789.0, p = 0.750
- Page's L, decreasing with first kernel size: L = 811.0, p = 0.250
- Spearman ρ vs first kernel size: -0.024, p = 0.789, n = 128

**Macro F1 -- adjacent paired Wilcoxon** (n pairs = folds shared by both variants; Holm-corrected within this arm)

| Comparison | Δ mean (pp) | Δ median (pp) | n | p | p (Holm) | sig. |
|---|---|---|---|---|---|---|
| `d3_k3` → `d3_k7` | +2.15 | +1.62 | 32 | 0.021 | 0.062 | no |
| `d3_k7` → `d3_k11` | -1.56 | +0.04 | 32 | 0.432 | 0.432 | no |
| `d3_k11` → `d3_k64` | -1.93 | -2.04 | 32 | 0.095 | 0.189 | no |

**Macro F1 -- trend across first kernel size** (Page's L over 32 folds x 4 variants)

- Page's L, increasing with first kernel size: L = 788.0, p = 0.769
- Page's L, decreasing with first kernel size: L = 812.0, p = 0.231
- Spearman ρ vs first kernel size: -0.036, p = 0.690, n = 128

### AlexNet

#### R2.2 -- depth ablation

| Variant | Depth | Params | RF (samples) | B.Acc | Macro F1 | B.Acc (std over rounds) |
|---|---|---|---|---|---|---|
| `d1_k11` | 1 | 1,448,708 | 19 | 69.16 ± 19.01 | 65.18 ± 22.21 | 69.16 ± 3.06 |
| `d2_k11` | 2 | 2,296,772 | 67 | 70.47 ± 20.23 | 66.82 ± 23.51 | 70.47 ± 4.80 |
| `d3_k11` | 3 | 3,697,988 | 131 | 70.64 ± 18.53 | 67.30 ± 21.55 | 70.64 ± 5.23 |
| `d5_k11` | 5 | 3,403,588 | 195 | 71.44 ± 18.11 | 67.62 ± 21.31 | 71.44 ± 3.31 |

**Balanced Accuracy -- adjacent paired Wilcoxon** (n pairs = folds shared by both variants; Holm-corrected within this arm)

| Comparison | Δ mean (pp) | Δ median (pp) | n | p | p (Holm) | sig. |
|---|---|---|---|---|---|---|
| `d1_k11` → `d2_k11` | +1.31 | +1.32 | 32 | 0.119 | 0.358 | no |
| `d2_k11` → `d3_k11` | +0.17 | -0.86 | 32 | 0.832 | 1.000 | no |
| `d3_k11` → `d5_k11` | +0.80 | +0.40 | 32 | 0.561 | 1.000 | no |

**Balanced Accuracy -- trend across conv layers** (Page's L over 32 folds x 4 variants)

- Page's L, increasing with conv layers: L = 827.0, p = 0.049
- Page's L, decreasing with conv layers: L = 773.0, p = 0.951
- Spearman ρ vs conv layers: +0.038, p = 0.669, n = 128

**Macro F1 -- adjacent paired Wilcoxon** (n pairs = folds shared by both variants; Holm-corrected within this arm)

| Comparison | Δ mean (pp) | Δ median (pp) | n | p | p (Holm) | sig. |
|---|---|---|---|---|---|---|
| `d1_k11` → `d2_k11` | +1.64 | +1.70 | 32 | 0.139 | 0.416 | no |
| `d2_k11` → `d3_k11` | +0.48 | -0.27 | 32 | 0.875 | 1.000 | no |
| `d3_k11` → `d5_k11` | +0.31 | -0.13 | 32 | 0.890 | 1.000 | no |

**Macro F1 -- trend across conv layers** (Page's L over 32 folds x 4 variants)

- Page's L, increasing with conv layers: L = 828.0, p = 0.043
- Page's L, decreasing with conv layers: L = 772.0, p = 0.957
- Spearman ρ vs conv layers: +0.035, p = 0.697, n = 128

#### R2.3 -- receptive field ablation

| Variant | First kernel | Params | RF (samples) | B.Acc | Macro F1 | B.Acc (std over rounds) |
|---|---|---|---|---|---|---|
| `d5_k3` | 3 | 3,403,076 | 187 | 74.25 ± 17.35 | 71.27 ± 20.48 | 74.25 ± 2.50 |
| `d5_k7` | 7 | 3,403,332 | 191 | 71.35 ± 16.63 | 67.68 ± 19.82 | 71.35 ± 3.98 |
| `d5_k11` | 11 | 3,403,588 | 195 | 71.44 ± 18.11 | 67.62 ± 21.31 | 71.44 ± 3.31 |
| `d5_k64` | 64 | 3,406,980 | 248 | 69.05 ± 19.77 | 64.95 ± 22.58 | 69.05 ± 7.36 |

**Balanced Accuracy -- adjacent paired Wilcoxon** (n pairs = folds shared by both variants; Holm-corrected within this arm)

| Comparison | Δ mean (pp) | Δ median (pp) | n | p | p (Holm) | sig. |
|---|---|---|---|---|---|---|
| `d5_k3` → `d5_k7` | -2.90 | -2.71 | 32 | 0.003 | 0.008 | yes |
| `d5_k7` → `d5_k11` | +0.09 | -0.16 | 32 | 0.993 | 0.993 | no |
| `d5_k11` → `d5_k64` | -2.38 | -0.14 | 32 | 0.270 | 0.540 | no |

**Balanced Accuracy -- trend across first kernel size** (Page's L over 32 folds x 4 variants)

- Page's L, increasing with first kernel size: L = 759.0, p = 0.994
- Page's L, decreasing with first kernel size: L = 841.0, p = 0.006
- Spearman ρ vs first kernel size: -0.095, p = 0.288, n = 128

**Macro F1 -- adjacent paired Wilcoxon** (n pairs = folds shared by both variants; Holm-corrected within this arm)

| Comparison | Δ mean (pp) | Δ median (pp) | n | p | p (Holm) | sig. |
|---|---|---|---|---|---|---|
| `d5_k3` → `d5_k7` | -3.59 | -3.54 | 32 | 0.002 | 0.005 | yes |
| `d5_k7` → `d5_k11` | -0.06 | -0.13 | 32 | 0.949 | 0.949 | no |
| `d5_k11` → `d5_k64` | -2.67 | -0.09 | 32 | 0.224 | 0.448 | no |

**Macro F1 -- trend across first kernel size** (Page's L over 32 folds x 4 variants)

- Page's L, increasing with first kernel size: L = 754.0, p = 0.998
- Page's L, decreasing with first kernel size: L = 846.0, p = 0.002
- Spearman ρ vs first kernel size: -0.098, p = 0.272, n = 128

### ResNet-18

#### R2.2 -- depth ablation

| Variant | Depth | Params | RF (samples) | B.Acc | Macro F1 | B.Acc (std over rounds) |
|---|---|---|---|---|---|---|
| `d10_k7` | 10 | 1,753,156 | 195 | 61.38 ± 20.62 | 56.21 ± 23.63 | 61.38 ± 5.42 |
| `d18_k7` | 18 | 3,845,956 | 435 | 63.21 ± 20.97 | 57.36 ± 25.55 | 63.21 ± 5.14 |
| `d26_k7` | 26 | 5,938,756 | 675 | 64.20 ± 19.26 | 58.93 ± 23.04 | 64.20 ± 7.11 |
| `d34_k7` | 34 | 8,031,556 | 915 | 64.62 ± 20.64 | 59.40 ± 24.38 | 64.62 ± 4.96 |

**Balanced Accuracy -- adjacent paired Wilcoxon** (n pairs = folds shared by both variants; Holm-corrected within this arm)

| Comparison | Δ mean (pp) | Δ median (pp) | n | p | p (Holm) | sig. |
|---|---|---|---|---|---|---|
| `d10_k7` → `d18_k7` | +1.83 | +1.36 | 32 | 0.111 | 0.332 | no |
| `d18_k7` → `d26_k7` | +1.00 | +1.36 | 32 | 0.231 | 0.462 | no |
| `d26_k7` → `d34_k7` | +0.42 | +0.33 | 32 | 0.378 | 0.462 | no |

**Balanced Accuracy -- trend across weighted layers** (Page's L over 32 folds x 4 variants)

- Page's L, increasing with weighted layers: L = 866.5, p = <0.001
- Page's L, decreasing with weighted layers: L = 733.5, p = 1.000
- Spearman ρ vs weighted layers: +0.055, p = 0.541, n = 128

**Macro F1 -- adjacent paired Wilcoxon** (n pairs = folds shared by both variants; Holm-corrected within this arm)

| Comparison | Δ mean (pp) | Δ median (pp) | n | p | p (Holm) | sig. |
|---|---|---|---|---|---|---|
| `d10_k7` → `d18_k7` | +1.16 | +0.54 | 32 | 0.421 | 0.759 | no |
| `d18_k7` → `d26_k7` | +1.56 | +0.53 | 32 | 0.379 | 0.759 | no |
| `d26_k7` → `d34_k7` | +0.47 | +1.20 | 32 | 0.239 | 0.716 | no |

**Macro F1 -- trend across weighted layers** (Page's L over 32 folds x 4 variants)

- Page's L, increasing with weighted layers: L = 843.0, p = 0.004
- Page's L, decreasing with weighted layers: L = 757.0, p = 0.996
- Spearman ρ vs weighted layers: +0.058, p = 0.515, n = 128

#### R2.3 -- receptive field ablation

| Variant | First kernel | Params | RF (samples) | B.Acc | Macro F1 | B.Acc (std over rounds) |
|---|---|---|---|---|---|---|
| `d18_k3` | 3 | 3,845,700 | 431 | 63.12 ± 20.27 | 56.52 ± 23.95 | 63.12 ± 4.08 |
| `d18_k7` | 7 | 3,845,956 | 435 | 63.21 ± 20.97 | 57.36 ± 25.55 | 63.21 ± 5.14 |
| `d18_k11` | 11 | 3,846,212 | 439 | 63.90 ± 20.22 | 58.19 ± 23.86 | 63.90 ± 5.40 |
| `d18_k64` | 64 | 3,849,604 | 492 | 63.90 ± 20.83 | 58.72 ± 24.88 | 63.90 ± 4.65 |

**Balanced Accuracy -- adjacent paired Wilcoxon** (n pairs = folds shared by both variants; Holm-corrected within this arm)

| Comparison | Δ mean (pp) | Δ median (pp) | n | p | p (Holm) | sig. |
|---|---|---|---|---|---|---|
| `d18_k3` → `d18_k7` | +0.08 | -1.64 | 32 | 0.357 | 1.000 | no |
| `d18_k7` → `d18_k11` | +0.69 | +0.48 | 32 | 0.347 | 1.000 | no |
| `d18_k11` → `d18_k64` | -0.00 | +0.35 | 32 | 0.719 | 1.000 | no |

**Balanced Accuracy -- trend across first kernel size** (Page's L over 32 folds x 4 variants)

- Page's L, increasing with first kernel size: L = 811.0, p = 0.250
- Page's L, decreasing with first kernel size: L = 789.0, p = 0.750
- Spearman ρ vs first kernel size: +0.014, p = 0.877, n = 128

**Macro F1 -- adjacent paired Wilcoxon** (n pairs = folds shared by both variants; Holm-corrected within this arm)

| Comparison | Δ mean (pp) | Δ median (pp) | n | p | p (Holm) | sig. |
|---|---|---|---|---|---|---|
| `d18_k3` → `d18_k7` | +0.84 | -0.16 | 32 | 0.934 | 1.000 | no |
| `d18_k7` → `d18_k11` | +0.83 | +0.71 | 32 | 0.350 | 1.000 | no |
| `d18_k11` → `d18_k64` | +0.53 | +0.49 | 32 | 0.512 | 1.000 | no |

**Macro F1 -- trend across first kernel size** (Page's L over 32 folds x 4 variants)

- Page's L, increasing with first kernel size: L = 821.0, p = 0.099
- Page's L, decreasing with first kernel size: L = 779.0, p = 0.901
- Spearman ρ vs first kernel size: +0.036, p = 0.683, n = 128

## CWRU12k

### Cross-model summary -- R2.2 -- depth ablation

| Model | Axis | B.Acc by level | Best | ρ B.Acc | Page p ↑ / ↓ (B.Acc) | ρ F1 | Page p ↑ / ↓ (F1) |
|---|---|---|---|---|---|---|---|
| 1D-CNN | conv blocks | 1: 71.99 · 2: 84.22 · 3: 88.60 · 5: 90.62 | `d5_k3` | +0.727 | <0.001 / 1.000 | +0.715 | <0.001 / 1.000 |
| AlexNet | conv layers | 1: 74.56 · 2: 86.43 · 3: 86.48 · 5: 84.05 | `d3_k11` | +0.423 | <0.001 / 1.000 | +0.427 | <0.001 / 1.000 |
| ResNet-18 | weighted layers | 10: 90.29 · 18: 90.30 · 26: 91.31 · 34: 90.65 | `d26_k7` | +0.067 | 0.129 / 0.871 | +0.070 | 0.164 / 0.836 |

### Cross-model summary -- R2.3 -- receptive field ablation

| Model | Axis | B.Acc by level | Best | ρ B.Acc | Page p ↑ / ↓ (B.Acc) | ρ F1 | Page p ↑ / ↓ (F1) |
|---|---|---|---|---|---|---|---|
| 1D-CNN | first kernel size | 3: 88.60 · 7: 89.39 · 11: 90.46 · 64: 88.88 | `d3_k11` | +0.037 | 0.171 / 0.829 | +0.041 | 0.089 / 0.911 |
| AlexNet | first kernel size | 3: 84.77 · 7: 86.02 · 11: 84.05 · 64: 85.07 | `d5_k7` | -0.002 | 0.291 / 0.709 | -0.024 | 0.250 / 0.750 |
| ResNet-18 | first kernel size | 3: 91.52 · 7: 90.30 · 11: 90.99 · 64: 88.61 | `d18_k3` | -0.143 | 1.000 / <0.001 | -0.146 | 1.000 / <0.001 |

B.Acc by level is the mean over all per-fold scores, in %. ρ is Spearman's against the arm's level; Page p ↑ tests a trend increasing with the level, ↓ a decreasing one. Each row is a separate within-model test.

### 1D-CNN

#### R2.2 -- depth ablation

| Variant | Depth | Params | RF (samples) | B.Acc | Macro F1 | B.Acc (std over rounds) |
|---|---|---|---|---|---|---|
| `d1_k3` | 1 | 33,508 | 3 | 71.99 ± 3.64 | 71.56 ± 3.73 | 71.99 ± 1.32 |
| `d2_k3` | 2 | 67,908 | 8 | 84.22 ± 5.14 | 84.16 ± 5.21 | 84.22 ± 1.46 |
| `d3_k3` | 3 | 139,780 | 18 | 88.60 ± 5.91 | 88.29 ± 6.14 | 88.60 ± 1.30 |
| `d5_k3` | 5 | 657,028 | 78 | 90.62 ± 7.58 | 90.32 ± 7.89 | 90.62 ± 1.30 |

**Balanced Accuracy -- adjacent paired Wilcoxon** (n pairs = folds shared by both variants; Holm-corrected within this arm)

| Comparison | Δ mean (pp) | Δ median (pp) | n | p | p (Holm) | sig. |
|---|---|---|---|---|---|---|
| `d1_k3` → `d2_k3` | +12.22 | +11.62 | 32 | <0.001 | <0.001 | yes |
| `d2_k3` → `d3_k3` | +4.39 | +3.81 | 32 | <0.001 | <0.001 | yes |
| `d3_k3` → `d5_k3` | +2.02 | +2.20 | 32 | 0.001 | 0.001 | yes |

**Balanced Accuracy -- trend across conv blocks** (Page's L over 32 folds x 4 variants)

- Page's L, increasing with conv blocks: L = 935.0, p = <0.001
- Page's L, decreasing with conv blocks: L = 665.0, p = 1.000
- Spearman ρ vs conv blocks: +0.727, p = <0.001, n = 128

**Macro F1 -- adjacent paired Wilcoxon** (n pairs = folds shared by both variants; Holm-corrected within this arm)

| Comparison | Δ mean (pp) | Δ median (pp) | n | p | p (Holm) | sig. |
|---|---|---|---|---|---|---|
| `d1_k3` → `d2_k3` | +12.60 | +12.19 | 32 | <0.001 | <0.001 | yes |
| `d2_k3` → `d3_k3` | +4.13 | +3.70 | 32 | <0.001 | <0.001 | yes |
| `d3_k3` → `d5_k3` | +2.03 | +2.66 | 32 | 0.002 | 0.002 | yes |

**Macro F1 -- trend across conv blocks** (Page's L over 32 folds x 4 variants)

- Page's L, increasing with conv blocks: L = 934.0, p = <0.001
- Page's L, decreasing with conv blocks: L = 666.0, p = 1.000
- Spearman ρ vs conv blocks: +0.715, p = <0.001, n = 128

#### R2.3 -- receptive field ablation

| Variant | First kernel | Params | RF (samples) | B.Acc | Macro F1 | B.Acc (std over rounds) |
|---|---|---|---|---|---|---|
| `d3_k3` | 3 | 139,780 | 18 | 88.60 ± 5.91 | 88.29 ± 6.14 | 88.60 ± 1.30 |
| `d3_k7` | 7 | 139,844 | 22 | 89.39 ± 6.69 | 88.97 ± 7.22 | 89.39 ± 1.73 |
| `d3_k11` | 11 | 139,908 | 26 | 90.46 ± 5.57 | 90.19 ± 5.76 | 90.46 ± 0.97 |
| `d3_k64` | 64 | 140,756 | 79 | 88.88 ± 6.26 | 88.64 ± 6.45 | 88.88 ± 1.37 |

**Balanced Accuracy -- adjacent paired Wilcoxon** (n pairs = folds shared by both variants; Holm-corrected within this arm)

| Comparison | Δ mean (pp) | Δ median (pp) | n | p | p (Holm) | sig. |
|---|---|---|---|---|---|---|
| `d3_k3` → `d3_k7` | +0.79 | +0.72 | 32 | 0.018 | 0.035 | yes |
| `d3_k7` → `d3_k11` | +1.07 | +0.24 | 32 | 0.142 | 0.142 | no |
| `d3_k11` → `d3_k64` | -1.58 | -1.58 | 32 | 0.003 | 0.009 | yes |

**Balanced Accuracy -- trend across first kernel size** (Page's L over 32 folds x 4 variants)

- Page's L, increasing with first kernel size: L = 815.5, p = 0.171
- Page's L, decreasing with first kernel size: L = 784.5, p = 0.829
- Spearman ρ vs first kernel size: +0.037, p = 0.679, n = 128

**Macro F1 -- adjacent paired Wilcoxon** (n pairs = folds shared by both variants; Holm-corrected within this arm)

| Comparison | Δ mean (pp) | Δ median (pp) | n | p | p (Holm) | sig. |
|---|---|---|---|---|---|---|
| `d3_k3` → `d3_k7` | +0.68 | +0.73 | 32 | 0.034 | 0.068 | no |
| `d3_k7` → `d3_k11` | +1.22 | +0.24 | 32 | 0.178 | 0.178 | no |
| `d3_k11` → `d3_k64` | -1.55 | -1.65 | 32 | 0.004 | 0.012 | yes |

**Macro F1 -- trend across first kernel size** (Page's L over 32 folds x 4 variants)

- Page's L, increasing with first kernel size: L = 822.0, p = 0.089
- Page's L, decreasing with first kernel size: L = 778.0, p = 0.911
- Spearman ρ vs first kernel size: +0.041, p = 0.643, n = 128

### AlexNet

#### R2.2 -- depth ablation

| Variant | Depth | Params | RF (samples) | B.Acc | Macro F1 | B.Acc (std over rounds) |
|---|---|---|---|---|---|---|
| `d1_k11` | 1 | 1,448,708 | 19 | 74.56 ± 4.65 | 74.34 ± 4.91 | 74.56 ± 0.89 |
| `d2_k11` | 2 | 2,296,772 | 67 | 86.43 ± 5.52 | 86.04 ± 5.85 | 86.43 ± 0.78 |
| `d3_k11` | 3 | 3,697,988 | 131 | 86.48 ± 5.82 | 86.09 ± 5.85 | 86.48 ± 1.36 |
| `d5_k11` | 5 | 3,403,588 | 195 | 84.05 ± 5.27 | 83.94 ± 4.87 | 84.05 ± 2.04 |

**Balanced Accuracy -- adjacent paired Wilcoxon** (n pairs = folds shared by both variants; Holm-corrected within this arm)

| Comparison | Δ mean (pp) | Δ median (pp) | n | p | p (Holm) | sig. |
|---|---|---|---|---|---|---|
| `d1_k11` → `d2_k11` | +11.87 | +11.64 | 32 | <0.001 | <0.001 | yes |
| `d2_k11` → `d3_k11` | +0.04 | -0.49 | 32 | 0.905 | 0.905 | no |
| `d3_k11` → `d5_k11` | -2.43 | -1.17 | 32 | 0.019 | 0.039 | yes |

**Balanced Accuracy -- trend across conv layers** (Page's L over 32 folds x 4 variants)

- Page's L, increasing with conv layers: L = 865.0, p = <0.001
- Page's L, decreasing with conv layers: L = 735.0, p = 1.000
- Spearman ρ vs conv layers: +0.423, p = <0.001, n = 128

**Macro F1 -- adjacent paired Wilcoxon** (n pairs = folds shared by both variants; Holm-corrected within this arm)

| Comparison | Δ mean (pp) | Δ median (pp) | n | p | p (Holm) | sig. |
|---|---|---|---|---|---|---|
| `d1_k11` → `d2_k11` | +11.69 | +11.67 | 32 | <0.001 | <0.001 | yes |
| `d2_k11` → `d3_k11` | +0.05 | -0.41 | 32 | 0.978 | 0.978 | no |
| `d3_k11` → `d5_k11` | -2.14 | -1.06 | 32 | 0.036 | 0.072 | no |

**Macro F1 -- trend across conv layers** (Page's L over 32 folds x 4 variants)

- Page's L, increasing with conv layers: L = 872.0, p = <0.001
- Page's L, decreasing with conv layers: L = 728.0, p = 1.000
- Spearman ρ vs conv layers: +0.427, p = <0.001, n = 128

#### R2.3 -- receptive field ablation

| Variant | First kernel | Params | RF (samples) | B.Acc | Macro F1 | B.Acc (std over rounds) |
|---|---|---|---|---|---|---|
| `d5_k3` | 3 | 3,403,076 | 187 | 84.77 ± 4.24 | 84.42 ± 4.46 | 84.77 ± 1.48 |
| `d5_k7` | 7 | 3,403,332 | 191 | 86.02 ± 4.27 | 85.87 ± 4.31 | 86.02 ± 2.09 |
| `d5_k11` | 11 | 3,403,588 | 195 | 84.05 ± 5.27 | 83.94 ± 4.87 | 84.05 ± 2.04 |
| `d5_k64` | 64 | 3,406,980 | 248 | 85.07 ± 7.34 | 84.40 ± 7.93 | 85.07 ± 1.39 |

**Balanced Accuracy -- adjacent paired Wilcoxon** (n pairs = folds shared by both variants; Holm-corrected within this arm)

| Comparison | Δ mean (pp) | Δ median (pp) | n | p | p (Holm) | sig. |
|---|---|---|---|---|---|---|
| `d5_k3` → `d5_k7` | +1.25 | +1.09 | 32 | 0.080 | 0.241 | no |
| `d5_k7` → `d5_k11` | -1.98 | -1.19 | 32 | 0.154 | 0.309 | no |
| `d5_k11` → `d5_k64` | +1.03 | +1.12 | 32 | 0.625 | 0.625 | no |

**Balanced Accuracy -- trend across first kernel size** (Page's L over 32 folds x 4 variants)

- Page's L, increasing with first kernel size: L = 809.0, p = 0.291
- Page's L, decreasing with first kernel size: L = 791.0, p = 0.709
- Spearman ρ vs first kernel size: -0.002, p = 0.981, n = 128

**Macro F1 -- adjacent paired Wilcoxon** (n pairs = folds shared by both variants; Holm-corrected within this arm)

| Comparison | Δ mean (pp) | Δ median (pp) | n | p | p (Holm) | sig. |
|---|---|---|---|---|---|---|
| `d5_k3` → `d5_k7` | +1.45 | +1.06 | 32 | 0.071 | 0.212 | no |
| `d5_k7` → `d5_k11` | -1.92 | -1.19 | 32 | 0.115 | 0.230 | no |
| `d5_k11` → `d5_k64` | +0.45 | +0.21 | 32 | 0.818 | 0.818 | no |

**Macro F1 -- trend across first kernel size** (Page's L over 32 folds x 4 variants)

- Page's L, increasing with first kernel size: L = 811.0, p = 0.250
- Page's L, decreasing with first kernel size: L = 789.0, p = 0.750
- Spearman ρ vs first kernel size: -0.024, p = 0.786, n = 128

### ResNet-18

#### R2.2 -- depth ablation

| Variant | Depth | Params | RF (samples) | B.Acc | Macro F1 | B.Acc (std over rounds) |
|---|---|---|---|---|---|---|
| `d10_k7` | 10 | 1,753,156 | 195 | 90.29 ± 6.36 | 89.62 ± 7.22 | 90.29 ± 1.32 |
| `d18_k7` | 18 | 3,845,956 | 435 | 90.30 ± 7.95 | 90.19 ± 8.09 | 90.30 ± 1.11 |
| `d26_k7` | 26 | 5,938,756 | 675 | 91.31 ± 6.76 | 91.02 ± 7.23 | 91.31 ± 1.81 |
| `d34_k7` | 34 | 8,031,556 | 915 | 90.65 ± 7.78 | 90.28 ± 8.38 | 90.65 ± 1.21 |

**Balanced Accuracy -- adjacent paired Wilcoxon** (n pairs = folds shared by both variants; Holm-corrected within this arm)

| Comparison | Δ mean (pp) | Δ median (pp) | n | p | p (Holm) | sig. |
|---|---|---|---|---|---|---|
| `d10_k7` → `d18_k7` | +0.01 | +0.59 | 32 | 0.477 | 0.477 | no |
| `d18_k7` → `d26_k7` | +1.01 | +0.52 | 32 | 0.128 | 0.255 | no |
| `d26_k7` → `d34_k7` | -0.66 | -0.44 | 32 | 0.050 | 0.150 | no |

**Balanced Accuracy -- trend across weighted layers** (Page's L over 32 folds x 4 variants)

- Page's L, increasing with weighted layers: L = 818.5, p = 0.129
- Page's L, decreasing with weighted layers: L = 781.5, p = 0.871
- Spearman ρ vs weighted layers: +0.067, p = 0.450, n = 128

**Macro F1 -- adjacent paired Wilcoxon** (n pairs = folds shared by both variants; Holm-corrected within this arm)

| Comparison | Δ mean (pp) | Δ median (pp) | n | p | p (Holm) | sig. |
|---|---|---|---|---|---|---|
| `d10_k7` → `d18_k7` | +0.57 | +0.58 | 32 | 0.322 | 0.322 | no |
| `d18_k7` → `d26_k7` | +0.84 | +0.50 | 32 | 0.134 | 0.267 | no |
| `d26_k7` → `d34_k7` | -0.75 | -0.50 | 32 | 0.068 | 0.204 | no |

**Macro F1 -- trend across weighted layers** (Page's L over 32 folds x 4 variants)

- Page's L, increasing with weighted layers: L = 816.0, p = 0.164
- Page's L, decreasing with weighted layers: L = 784.0, p = 0.836
- Spearman ρ vs weighted layers: +0.070, p = 0.431, n = 128

#### R2.3 -- receptive field ablation

| Variant | First kernel | Params | RF (samples) | B.Acc | Macro F1 | B.Acc (std over rounds) |
|---|---|---|---|---|---|---|
| `d18_k3` | 3 | 3,845,700 | 431 | 91.52 ± 6.77 | 91.09 ± 7.29 | 91.52 ± 1.48 |
| `d18_k7` | 7 | 3,845,956 | 435 | 90.30 ± 7.95 | 90.19 ± 8.09 | 90.30 ± 1.11 |
| `d18_k11` | 11 | 3,846,212 | 439 | 90.99 ± 8.03 | 90.71 ± 8.34 | 90.99 ± 2.10 |
| `d18_k64` | 64 | 3,849,604 | 492 | 88.61 ± 7.22 | 88.10 ± 7.80 | 88.61 ± 1.78 |

**Balanced Accuracy -- adjacent paired Wilcoxon** (n pairs = folds shared by both variants; Holm-corrected within this arm)

| Comparison | Δ mean (pp) | Δ median (pp) | n | p | p (Holm) | sig. |
|---|---|---|---|---|---|---|
| `d18_k3` → `d18_k7` | -1.22 | -0.75 | 32 | 0.129 | 0.258 | no |
| `d18_k7` → `d18_k11` | +0.68 | +0.39 | 32 | 0.149 | 0.258 | no |
| `d18_k11` → `d18_k64` | -2.38 | -2.29 | 32 | 0.008 | 0.023 | yes |

**Balanced Accuracy -- trend across first kernel size** (Page's L over 32 folds x 4 variants)

- Page's L, increasing with first kernel size: L = 740.5, p = 1.000
- Page's L, decreasing with first kernel size: L = 859.5, p = <0.001
- Spearman ρ vs first kernel size: -0.143, p = 0.106, n = 128

**Macro F1 -- adjacent paired Wilcoxon** (n pairs = folds shared by both variants; Holm-corrected within this arm)

| Comparison | Δ mean (pp) | Δ median (pp) | n | p | p (Holm) | sig. |
|---|---|---|---|---|---|---|
| `d18_k3` → `d18_k7` | -0.90 | -0.75 | 32 | 0.139 | 0.277 | no |
| `d18_k7` → `d18_k11` | +0.52 | +0.42 | 32 | 0.154 | 0.277 | no |
| `d18_k11` → `d18_k64` | -2.60 | -2.37 | 32 | 0.003 | 0.009 | yes |

**Macro F1 -- trend across first kernel size** (Page's L over 32 folds x 4 variants)

- Page's L, increasing with first kernel size: L = 737.0, p = 1.000
- Page's L, decreasing with first kernel size: L = 863.0, p = <0.001
- Spearman ρ vs first kernel size: -0.146, p = 0.100, n = 128

## CWRU48k

### Cross-model summary -- R2.2 -- depth ablation

| Model | Axis | B.Acc by level | Best | ρ B.Acc | Page p ↑ / ↓ (B.Acc) | ρ F1 | Page p ↑ / ↓ (F1) |
|---|---|---|---|---|---|---|---|
| 1D-CNN | conv blocks | 1: 51.73 · 2: 61.99 · 3: 82.83 · 5: 87.18 | `d5_k3` | +0.834 | <0.001 / 1.000 | +0.837 | <0.001 / 1.000 |
| AlexNet | conv layers | 1: 59.87 · 2: 70.53 · 3: 71.86 · 5: 70.18 | `d3_k11` | +0.331 | <0.001 / 1.000 | +0.327 | <0.001 / 1.000 |
| ResNet-18 | weighted layers | 10: 87.39 · 18: 80.79 · 26: 82.65 · 34: 83.90 | `d10_k7` | -0.063 | 0.954 / 0.046 | -0.083 | 0.978 / 0.022 |

### Cross-model summary -- R2.3 -- receptive field ablation

| Model | Axis | B.Acc by level | Best | ρ B.Acc | Page p ↑ / ↓ (B.Acc) | ρ F1 | Page p ↑ / ↓ (F1) |
|---|---|---|---|---|---|---|---|
| 1D-CNN | first kernel size | 3: 82.83 · 7: 82.95 · 11: 85.13 · 64: 90.07 | `d3_k64` | +0.262 | <0.001 / 1.000 | +0.281 | <0.001 / 1.000 |
| AlexNet | first kernel size | 3: 66.70 · 7: 69.24 · 11: 70.18 · 64: 85.81 | `d5_k64` | +0.555 | <0.001 / 1.000 | +0.546 | <0.001 / 1.000 |
| ResNet-18 | first kernel size | 3: 81.51 · 7: 80.79 · 11: 83.10 · 64: 86.65 | `d18_k64` | +0.143 | 0.002 / 0.998 | +0.154 | 0.001 / 0.999 |

B.Acc by level is the mean over all per-fold scores, in %. ρ is Spearman's against the arm's level; Page p ↑ tests a trend increasing with the level, ↓ a decreasing one. Each row is a separate within-model test.

### 1D-CNN

#### R2.2 -- depth ablation

| Variant | Depth | Params | RF (samples) | B.Acc | Macro F1 | B.Acc (std over rounds) |
|---|---|---|---|---|---|---|
| `d1_k3` | 1 | 33,379 | 3 | 51.73 ± 8.09 | 48.65 ± 7.46 | 51.73 ± 1.16 |
| `d2_k3` | 2 | 67,779 | 8 | 61.99 ± 9.36 | 60.61 ± 10.57 | 61.99 ± 3.24 |
| `d3_k3` | 3 | 139,651 | 18 | 82.83 ± 8.97 | 82.67 ± 8.78 | 82.83 ± 1.47 |
| `d5_k3` | 5 | 656,899 | 78 | 87.18 ± 9.32 | 86.89 ± 10.13 | 87.18 ± 2.24 |

**Balanced Accuracy -- adjacent paired Wilcoxon** (n pairs = folds shared by both variants; Holm-corrected within this arm)

| Comparison | Δ mean (pp) | Δ median (pp) | n | p | p (Holm) | sig. |
|---|---|---|---|---|---|---|
| `d1_k3` → `d2_k3` | +10.26 | +10.28 | 32 | <0.001 | <0.001 | yes |
| `d2_k3` → `d3_k3` | +20.84 | +20.69 | 32 | <0.001 | <0.001 | yes |
| `d3_k3` → `d5_k3` | +4.35 | +5.03 | 32 | <0.001 | <0.001 | yes |

**Balanced Accuracy -- trend across conv blocks** (Page's L over 32 folds x 4 variants)

- Page's L, increasing with conv blocks: L = 946.0, p = <0.001
- Page's L, decreasing with conv blocks: L = 654.0, p = 1.000
- Spearman ρ vs conv blocks: +0.834, p = <0.001, n = 128

**Macro F1 -- adjacent paired Wilcoxon** (n pairs = folds shared by both variants; Holm-corrected within this arm)

| Comparison | Δ mean (pp) | Δ median (pp) | n | p | p (Holm) | sig. |
|---|---|---|---|---|---|---|
| `d1_k3` → `d2_k3` | +11.96 | +14.47 | 32 | <0.001 | <0.001 | yes |
| `d2_k3` → `d3_k3` | +22.06 | +22.64 | 32 | <0.001 | <0.001 | yes |
| `d3_k3` → `d5_k3` | +4.21 | +4.94 | 32 | 0.001 | 0.001 | yes |

**Macro F1 -- trend across conv blocks** (Page's L over 32 folds x 4 variants)

- Page's L, increasing with conv blocks: L = 947.0, p = <0.001
- Page's L, decreasing with conv blocks: L = 653.0, p = 1.000
- Spearman ρ vs conv blocks: +0.837, p = <0.001, n = 128

#### R2.3 -- receptive field ablation

| Variant | First kernel | Params | RF (samples) | B.Acc | Macro F1 | B.Acc (std over rounds) |
|---|---|---|---|---|---|---|
| `d3_k3` | 3 | 139,651 | 18 | 82.83 ± 8.97 | 82.67 ± 8.78 | 82.83 ± 1.47 |
| `d3_k7` | 7 | 139,715 | 22 | 82.95 ± 10.96 | 82.50 ± 11.29 | 82.95 ± 0.78 |
| `d3_k11` | 11 | 139,779 | 26 | 85.13 ± 9.37 | 85.15 ± 9.34 | 85.13 ± 1.88 |
| `d3_k64` | 64 | 140,627 | 79 | 90.07 ± 8.11 | 90.13 ± 8.49 | 90.07 ± 0.58 |

**Balanced Accuracy -- adjacent paired Wilcoxon** (n pairs = folds shared by both variants; Holm-corrected within this arm)

| Comparison | Δ mean (pp) | Δ median (pp) | n | p | p (Holm) | sig. |
|---|---|---|---|---|---|---|
| `d3_k3` → `d3_k7` | +0.12 | -0.00 | 32 | 0.791 | 0.791 | no |
| `d3_k7` → `d3_k11` | +2.18 | +2.50 | 32 | 0.014 | 0.028 | yes |
| `d3_k11` → `d3_k64` | +4.94 | +4.58 | 32 | <0.001 | <0.001 | yes |

**Balanced Accuracy -- trend across first kernel size** (Page's L over 32 folds x 4 variants)

- Page's L, increasing with first kernel size: L = 885.5, p = <0.001
- Page's L, decreasing with first kernel size: L = 714.5, p = 1.000
- Spearman ρ vs first kernel size: +0.262, p = 0.003, n = 128

**Macro F1 -- adjacent paired Wilcoxon** (n pairs = folds shared by both variants; Holm-corrected within this arm)

| Comparison | Δ mean (pp) | Δ median (pp) | n | p | p (Holm) | sig. |
|---|---|---|---|---|---|---|
| `d3_k3` → `d3_k7` | -0.17 | -0.05 | 32 | 0.963 | 0.963 | no |
| `d3_k7` → `d3_k11` | +2.65 | +2.39 | 32 | 0.014 | 0.028 | yes |
| `d3_k11` → `d3_k64` | +4.98 | +4.25 | 32 | <0.001 | <0.001 | yes |

**Macro F1 -- trend across first kernel size** (Page's L over 32 folds x 4 variants)

- Page's L, increasing with first kernel size: L = 883.0, p = <0.001
- Page's L, decreasing with first kernel size: L = 717.0, p = 1.000
- Spearman ρ vs first kernel size: +0.281, p = 0.001, n = 128

### AlexNet

#### R2.2 -- depth ablation

| Variant | Depth | Params | RF (samples) | B.Acc | Macro F1 | B.Acc (std over rounds) |
|---|---|---|---|---|---|---|
| `d1_k11` | 1 | 1,447,683 | 19 | 59.87 ± 10.02 | 57.82 ± 11.05 | 59.87 ± 1.92 |
| `d2_k11` | 2 | 2,295,747 | 67 | 70.53 ± 11.08 | 70.02 ± 11.61 | 70.53 ± 2.85 |
| `d3_k11` | 3 | 3,696,963 | 131 | 71.86 ± 12.26 | 71.59 ± 13.05 | 71.86 ± 2.43 |
| `d5_k11` | 5 | 3,402,563 | 195 | 70.18 ± 10.48 | 69.51 ± 11.61 | 70.18 ± 3.62 |

**Balanced Accuracy -- adjacent paired Wilcoxon** (n pairs = folds shared by both variants; Holm-corrected within this arm)

| Comparison | Δ mean (pp) | Δ median (pp) | n | p | p (Holm) | sig. |
|---|---|---|---|---|---|---|
| `d1_k11` → `d2_k11` | +10.66 | +12.78 | 32 | <0.001 | <0.001 | yes |
| `d2_k11` → `d3_k11` | +1.33 | +0.83 | 32 | 0.203 | 0.405 | no |
| `d3_k11` → `d5_k11` | -1.68 | -0.28 | 32 | 0.710 | 0.710 | no |

**Balanced Accuracy -- trend across conv layers** (Page's L over 32 folds x 4 variants)

- Page's L, increasing with conv layers: L = 867.5, p = <0.001
- Page's L, decreasing with conv layers: L = 732.5, p = 1.000
- Spearman ρ vs conv layers: +0.331, p = <0.001, n = 128

**Macro F1 -- adjacent paired Wilcoxon** (n pairs = folds shared by both variants; Holm-corrected within this arm)

| Comparison | Δ mean (pp) | Δ median (pp) | n | p | p (Holm) | sig. |
|---|---|---|---|---|---|---|
| `d1_k11` → `d2_k11` | +12.21 | +13.13 | 32 | <0.001 | <0.001 | yes |
| `d2_k11` → `d3_k11` | +1.57 | +1.44 | 32 | 0.134 | 0.267 | no |
| `d3_k11` → `d5_k11` | -2.09 | -0.55 | 32 | 0.586 | 0.586 | no |

**Macro F1 -- trend across conv layers** (Page's L over 32 folds x 4 variants)

- Page's L, increasing with conv layers: L = 865.0, p = <0.001
- Page's L, decreasing with conv layers: L = 735.0, p = 1.000
- Spearman ρ vs conv layers: +0.327, p = <0.001, n = 128

#### R2.3 -- receptive field ablation

| Variant | First kernel | Params | RF (samples) | B.Acc | Macro F1 | B.Acc (std over rounds) |
|---|---|---|---|---|---|---|
| `d5_k3` | 3 | 3,402,051 | 187 | 66.70 ± 8.70 | 65.55 ± 9.72 | 66.70 ± 3.50 |
| `d5_k7` | 7 | 3,402,307 | 191 | 69.24 ± 10.38 | 68.41 ± 11.46 | 69.24 ± 3.54 |
| `d5_k11` | 11 | 3,402,563 | 195 | 70.18 ± 10.48 | 69.51 ± 11.61 | 70.18 ± 3.62 |
| `d5_k64` | 64 | 3,405,955 | 248 | 85.81 ± 10.42 | 85.32 ± 11.24 | 85.81 ± 4.18 |

**Balanced Accuracy -- adjacent paired Wilcoxon** (n pairs = folds shared by both variants; Holm-corrected within this arm)

| Comparison | Δ mean (pp) | Δ median (pp) | n | p | p (Holm) | sig. |
|---|---|---|---|---|---|---|
| `d5_k3` → `d5_k7` | +2.54 | +3.89 | 32 | 0.087 | 0.174 | no |
| `d5_k7` → `d5_k11` | +0.93 | -0.14 | 32 | 0.556 | 0.556 | no |
| `d5_k11` → `d5_k64` | +15.63 | +16.11 | 32 | <0.001 | <0.001 | yes |

**Balanced Accuracy -- trend across first kernel size** (Page's L over 32 folds x 4 variants)

- Page's L, increasing with first kernel size: L = 903.0, p = <0.001
- Page's L, decreasing with first kernel size: L = 697.0, p = 1.000
- Spearman ρ vs first kernel size: +0.555, p = <0.001, n = 128

**Macro F1 -- adjacent paired Wilcoxon** (n pairs = folds shared by both variants; Holm-corrected within this arm)

| Comparison | Δ mean (pp) | Δ median (pp) | n | p | p (Holm) | sig. |
|---|---|---|---|---|---|---|
| `d5_k3` → `d5_k7` | +2.86 | +3.61 | 32 | 0.087 | 0.174 | no |
| `d5_k7` → `d5_k11` | +1.09 | +1.16 | 32 | 0.548 | 0.548 | no |
| `d5_k11` → `d5_k64` | +15.81 | +16.26 | 32 | <0.001 | <0.001 | yes |

**Macro F1 -- trend across first kernel size** (Page's L over 32 folds x 4 variants)

- Page's L, increasing with first kernel size: L = 895.0, p = <0.001
- Page's L, decreasing with first kernel size: L = 705.0, p = 1.000
- Spearman ρ vs first kernel size: +0.546, p = <0.001, n = 128

### ResNet-18

#### R2.2 -- depth ablation

| Variant | Depth | Params | RF (samples) | B.Acc | Macro F1 | B.Acc (std over rounds) |
|---|---|---|---|---|---|---|
| `d10_k7` | 10 | 1,752,643 | 195 | 87.39 ± 9.83 | 87.67 ± 10.08 | 87.39 ± 3.44 |
| `d18_k7` | 18 | 3,845,443 | 435 | 80.79 ± 12.48 | 81.04 ± 12.53 | 80.79 ± 3.41 |
| `d26_k7` | 26 | 5,938,243 | 675 | 82.65 ± 14.05 | 81.80 ± 15.60 | 82.65 ± 3.75 |
| `d34_k7` | 34 | 8,031,043 | 915 | 83.90 ± 12.45 | 83.38 ± 13.57 | 83.90 ± 2.82 |

**Balanced Accuracy -- adjacent paired Wilcoxon** (n pairs = folds shared by both variants; Holm-corrected within this arm)

| Comparison | Δ mean (pp) | Δ median (pp) | n | p | p (Holm) | sig. |
|---|---|---|---|---|---|---|
| `d10_k7` → `d18_k7` | -6.59 | -5.56 | 32 | <0.001 | <0.001 | yes |
| `d18_k7` → `d26_k7` | +1.86 | -0.37 | 32 | 0.531 | 0.918 | no |
| `d26_k7` → `d34_k7` | +1.24 | +0.56 | 32 | 0.459 | 0.918 | no |

**Balanced Accuracy -- trend across weighted layers** (Page's L over 32 folds x 4 variants)

- Page's L, increasing with weighted layers: L = 772.5, p = 0.954
- Page's L, decreasing with weighted layers: L = 827.5, p = 0.046
- Spearman ρ vs weighted layers: -0.063, p = 0.478, n = 128

**Macro F1 -- adjacent paired Wilcoxon** (n pairs = folds shared by both variants; Holm-corrected within this arm)

| Comparison | Δ mean (pp) | Δ median (pp) | n | p | p (Holm) | sig. |
|---|---|---|---|---|---|---|
| `d10_k7` → `d18_k7` | -6.63 | -5.85 | 32 | <0.001 | 0.002 | yes |
| `d18_k7` → `d26_k7` | +0.75 | -0.49 | 32 | 0.877 | 0.931 | no |
| `d26_k7` → `d34_k7` | +1.59 | +1.18 | 32 | 0.465 | 0.931 | no |

**Macro F1 -- trend across weighted layers** (Page's L over 32 folds x 4 variants)

- Page's L, increasing with weighted layers: L = 767.0, p = 0.978
- Page's L, decreasing with weighted layers: L = 833.0, p = 0.022
- Spearman ρ vs weighted layers: -0.083, p = 0.354, n = 128

#### R2.3 -- receptive field ablation

| Variant | First kernel | Params | RF (samples) | B.Acc | Macro F1 | B.Acc (std over rounds) |
|---|---|---|---|---|---|---|
| `d18_k3` | 3 | 3,845,187 | 431 | 81.51 ± 14.70 | 81.15 ± 15.50 | 81.51 ± 3.82 |
| `d18_k7` | 7 | 3,845,443 | 435 | 80.79 ± 12.48 | 81.04 ± 12.53 | 80.79 ± 3.41 |
| `d18_k11` | 11 | 3,845,699 | 439 | 83.10 ± 13.91 | 83.17 ± 13.81 | 83.10 ± 3.48 |
| `d18_k64` | 64 | 3,849,091 | 492 | 86.65 ± 11.35 | 87.14 ± 11.33 | 86.65 ± 2.68 |

**Balanced Accuracy -- adjacent paired Wilcoxon** (n pairs = folds shared by both variants; Holm-corrected within this arm)

| Comparison | Δ mean (pp) | Δ median (pp) | n | p | p (Holm) | sig. |
|---|---|---|---|---|---|---|
| `d18_k3` → `d18_k7` | -0.71 | -0.83 | 32 | 0.634 | 0.634 | no |
| `d18_k7` → `d18_k11` | +2.30 | +1.81 | 32 | 0.181 | 0.362 | no |
| `d18_k11` → `d18_k64` | +3.55 | +3.61 | 32 | 0.044 | 0.133 | no |

**Balanced Accuracy -- trend across first kernel size** (Page's L over 32 folds x 4 variants)

- Page's L, increasing with first kernel size: L = 846.5, p = 0.002
- Page's L, decreasing with first kernel size: L = 753.5, p = 0.998
- Spearman ρ vs first kernel size: +0.143, p = 0.107, n = 128

**Macro F1 -- adjacent paired Wilcoxon** (n pairs = folds shared by both variants; Holm-corrected within this arm)

| Comparison | Δ mean (pp) | Δ median (pp) | n | p | p (Holm) | sig. |
|---|---|---|---|---|---|---|
| `d18_k3` → `d18_k7` | -0.11 | -0.70 | 32 | 0.766 | 0.766 | no |
| `d18_k7` → `d18_k11` | +2.13 | +1.70 | 32 | 0.232 | 0.464 | no |
| `d18_k11` → `d18_k64` | +3.97 | +3.63 | 32 | 0.043 | 0.130 | no |

**Macro F1 -- trend across first kernel size** (Page's L over 32 folds x 4 variants)

- Page's L, increasing with first kernel size: L = 849.5, p = 0.001
- Page's L, decreasing with first kernel size: L = 750.5, p = 0.999
- Spearman ρ vs first kernel size: +0.154, p = 0.083, n = 128
