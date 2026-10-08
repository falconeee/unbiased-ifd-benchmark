"""Aggregate the ablation result JSONs into the tables and tests the manuscript needs.

Reviewer #2's objection is about evidential support, so a table of means is not enough: the
claim "performance degrades (or does not degrade) with depth" has to survive a test. Three
things are computed per dataset, per model family and per arm:

``adjacent Wilcoxon``
    Paired signed-rank test between neighbouring variants (depth 1 vs 2, 2 vs 3, 3 vs 5;
    kernel 3 vs 7, 7 vs 11, 11 vs 64), pairing scores by ``(round, fold)``. The pairing is
    legitimate because every variant sees the identical fold assignment from
    ``src/fold_designs.json`` with the identical seed. Holm-corrected within each arm.

``Page's trend test``
    Ordered-alternative test over all variants of the arm at once, run in both directions,
    which is the test that actually addresses "monotonic in depth".

``Spearman``
    Effect direction and magnitude over the pooled per-fold scores.

Both dispersion conventions are reported side by side because they differ and the manuscript
is ambiguous about which it uses: ``fold`` (std over all 32 per-fold scores, what Table 4's
caption claims) and ``round`` (std over the 8 round means, what ``results.summary`` in the
result JSONs stores).

Each dataset section opens with a cross-model summary -- one row per family and arm, with the
per-level means and the trend statistics -- which is the table that answers whether a trend
seen in one architecture repeats in the others. No test is run *across* families: their
variants are different networks, so only the within-family trends are compared.
"""

from __future__ import annotations

import glob
import json
import os
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .registry import DATASET_NAMES, FAMILIES, STUDIES, arm_variants, document_family

METRICS = (("accuracy", "Balanced Accuracy"), ("f1_score", "Macro F1"))
ALPHA = 0.05


# --------------------------------------------------------------------------- loading
def load_results(output_dir: str) -> Dict[str, Dict[str, Dict[str, dict]]]:
    """``{dataset: {family: {variant: document}}}`` for every ablation JSON under ``output_dir``."""
    found: Dict[str, Dict[str, Dict[str, dict]]] = {}
    for ds_dir in sorted(glob.glob(os.path.join(output_dir, "*"))):
        if not os.path.isdir(ds_dir) or os.path.basename(ds_dir).startswith("_"):
            continue
        dataset = os.path.basename(ds_dir)
        for path in sorted(glob.glob(os.path.join(ds_dir, "*.json"))):
            if os.path.basename(path) == "_index.json":
                continue
            with open(path, encoding="utf-8") as fh:
                doc = json.load(fh)
            variant, family = doc.get("variant"), document_family(doc)
            if variant and family:
                found.setdefault(dataset, {}).setdefault(family, {})[variant] = doc
    return found


def fold_scores(doc: dict, metric: str) -> "OrderedDict[Tuple[int, int], float]":
    """``{(round, fold): score}`` across every executed round of one experiment."""
    scores: "OrderedDict[Tuple[int, int], float]" = OrderedDict()
    for round_block in doc.get("results", {}).get("rounds", []):
        rnd = round_block.get("round")
        for row in round_block.get("per_fold_metrics", []):
            value = row.get(metric)
            if value is not None:
                scores[(rnd, row["fold"])] = float(value)
    return scores


def round_means(doc: dict, metric: str) -> List[float]:
    means = []
    for round_block in doc.get("results", {}).get("rounds", []):
        values = [
            row[metric]
            for row in round_block.get("per_fold_metrics", [])
            if row.get(metric) is not None
        ]
        if values:
            means.append(float(np.mean(values)))
    return means


# ------------------------------------------------------------------------ statistics
def holm(p_values: Sequence[Optional[float]]) -> List[Optional[float]]:
    """Holm-Bonferroni step-down adjustment, preserving ``None`` entries."""
    indexed = [(p, i) for i, p in enumerate(p_values) if p is not None]
    adjusted: List[Optional[float]] = [None] * len(p_values)
    running = 0.0
    for rank, (p, i) in enumerate(sorted(indexed)):
        value = min(1.0, (len(indexed) - rank) * p)
        running = max(running, value)  # enforce monotonicity of the step-down sequence
        adjusted[i] = running
    return adjusted


def describe(doc: dict, metric: str) -> OrderedDict:
    folds = list(fold_scores(doc, metric).values())
    rounds = round_means(doc, metric)
    block = OrderedDict()
    block["n_folds"] = len(folds)
    block["n_rounds"] = len(rounds)
    block["mean"] = float(np.mean(folds)) if folds else None
    block["std_over_folds"] = float(np.std(folds, ddof=1)) if len(folds) > 1 else None
    block["std_over_rounds"] = float(np.std(rounds, ddof=1)) if len(rounds) > 1 else None
    block["min"] = float(np.min(folds)) if folds else None
    block["max"] = float(np.max(folds)) if folds else None
    return block


def paired_wilcoxon(doc_a: dict, doc_b: dict, metric: str) -> OrderedDict:
    """Signed-rank test on the folds both variants actually completed."""
    from scipy.stats import wilcoxon

    a, b = fold_scores(doc_a, metric), fold_scores(doc_b, metric)
    shared = [key for key in a if key in b]
    result = OrderedDict(n_pairs=len(shared))
    if len(shared) < 2:
        result["statistic"] = result["p_value"] = None
        result["median_difference"] = None
        result["note"] = "not enough shared folds"
        return result

    diff = np.array([b[key] - a[key] for key in shared])
    result["median_difference"] = float(np.median(diff))
    result["mean_difference"] = float(np.mean(diff))
    if np.allclose(diff, 0.0):
        result["statistic"] = result["p_value"] = None
        result["note"] = "identical scores on every shared fold"
        return result
    stat, p = wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
    result["statistic"] = float(stat)
    result["p_value"] = float(p)
    return result


def trend_tests(
    docs: Sequence[dict], levels: Sequence[float], metric: str
) -> OrderedDict:
    """Page's L in both directions plus Spearman over the pooled per-fold scores."""
    from scipy.stats import page_trend_test, spearmanr

    per_variant = [fold_scores(doc, metric) for doc in docs]
    shared = [key for key in per_variant[0] if all(key in s for s in per_variant[1:])]
    result = OrderedDict(n_blocks=len(shared), n_variants=len(docs))
    if len(shared) < 2 or len(docs) < 3:
        result["note"] = "not enough shared folds or variants for a trend test"
        return result

    # rows = folds (blocks), columns = variants in increasing level order
    matrix = np.array([[s[key] for s in per_variant] for key in shared])

    for direction, data in (("increasing", matrix), ("decreasing", matrix[:, ::-1])):
        try:
            page = page_trend_test(data, ranked=False)
            result[f"page_{direction}"] = OrderedDict(
                statistic=float(page.statistic), p_value=float(page.pvalue), method=page.method
            )
        except Exception as exc:  # noqa: BLE001 -- degenerate input, report instead of raising
            result[f"page_{direction}"] = OrderedDict(statistic=None, p_value=None, note=str(exc))

    x = np.repeat(np.asarray(levels, dtype=float), len(shared))
    y = matrix.T.ravel()
    rho, p = spearmanr(x, y)
    result["spearman"] = OrderedDict(
        rho=float(rho) if np.isfinite(rho) else None,
        p_value=float(p) if np.isfinite(p) else None,
        n=int(len(x)),
    )
    return result


# ----------------------------------------------------------------------------- arms
def arm_title(family: str, study: str) -> Tuple[str, str, str]:
    """``(title, axis, level label)`` of one arm of one family."""
    if study == "depth":
        return ("R2.2 -- depth ablation", FAMILIES[family].depth_unit, "Depth")
    return ("R2.3 -- receptive field ablation", "first kernel size", "First kernel")


def analyse_arm(available: Dict[str, dict], family: str, study: str) -> Optional[OrderedDict]:
    """Descriptives and tests for one arm of one family, or ``None`` below two variants."""
    wanted = arm_variants(family, study)
    present = [(key, level) for key, level in wanted if key in available]
    if len(present) < 2:
        return None
    docs = [available[key] for key, _ in present]
    levels = [level for _, level in present]

    arm: OrderedDict = OrderedDict()
    arm["variants"] = [
        OrderedDict(
            variant=key,
            level=level,
            depth=available[key]["architecture"]["depth"],
            first_kernel=available[key]["architecture"]["first_kernel"],
            num_parameters=available[key]["architecture"]["num_parameters"],
            receptive_field_samples=available[key]["architecture"][
                "receptive_field_samples"
            ],
            metrics=OrderedDict(
                (metric, describe(available[key], metric)) for metric, _ in METRICS
            ),
        )
        for key, level in present
    ]
    arm["complete"] = len(present) == len(wanted)
    arm["missing"] = [key for key, _ in wanted if key not in available]

    arm["comparisons"] = OrderedDict()
    arm["trend"] = OrderedDict()
    for metric, _label in METRICS:
        pairs = []
        for (key_a, _), (key_b, _) in zip(present, present[1:]):
            row = OrderedDict(baseline=key_a, candidate=key_b)
            row.update(paired_wilcoxon(available[key_a], available[key_b], metric))
            pairs.append(row)
        for row, adj in zip(pairs, holm([row.get("p_value") for row in pairs])):
            row["p_value_holm"] = adj
            row["significant_holm_0.05"] = (
                None if adj is None else bool(adj < ALPHA)
            )
        arm["comparisons"][metric] = pairs
        arm["trend"][metric] = trend_tests(docs, levels, metric)
    return arm


def analyse(results: Dict[str, Dict[str, Dict[str, dict]]]) -> OrderedDict:
    """``{dataset: {family: {study: arm}}}`` for every arm with at least two variants."""
    report: OrderedDict = OrderedDict()
    for dataset in DATASET_NAMES:
        by_family = results.get(dataset)
        if not by_family:
            continue
        ds_block: OrderedDict = OrderedDict()
        for family in FAMILIES:
            available = by_family.get(family)
            if not available:
                continue
            fam_block: OrderedDict = OrderedDict()
            for study in STUDIES:
                arm = analyse_arm(available, family, study)
                if arm is not None:
                    fam_block[study] = arm
            if fam_block:
                ds_block[family] = fam_block
        if ds_block:
            report[dataset] = ds_block
    return report


# ------------------------------------------------------------------------- rendering
def _pct(value: Optional[float]) -> str:
    return "--" if value is None else f"{100 * value:.2f}"


def _pm(stats: OrderedDict, dispersion: str = "std_over_folds") -> str:
    if stats.get("mean") is None:
        return "--"
    spread = stats.get(dispersion)
    if spread is None:
        return _pct(stats["mean"])
    return f"{_pct(stats['mean'])} ± {_pct(spread)}"


def _p(value: Optional[float]) -> str:
    if value is None:
        return "--"
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def _delta_pp(value: Optional[float]) -> str:
    """A difference of two rates, rendered in percentage points."""
    return "--" if value is None else f"{100 * value:+.2f}"


def _stat(block: Optional[dict], key: str, fmt: str) -> str:
    if not block or block.get(key) is None:
        return "--"
    return format(block[key], fmt)


def _page_p(trend: dict, direction: str) -> Optional[float]:
    block = trend.get(f"page_{direction}")
    return block.get("p_value") if block else None


def _best(arm: OrderedDict, metric: str = "accuracy") -> Optional[OrderedDict]:
    rows = [row for row in arm["variants"] if row["metrics"][metric]["mean"] is not None]
    return max(rows, key=lambda row: row["metrics"][metric]["mean"]) if rows else None


def _level_means(arm: OrderedDict, metric: str = "accuracy") -> str:
    return " · ".join(
        f"{int(row['level'])}: {_pct(row['metrics'][metric]['mean'])}"
        for row in arm["variants"]
    )


def render_summary_markdown(families: OrderedDict) -> List[str]:
    """One table per arm, one row per family: the cross-model reading of the ablation."""
    lines: List[str] = []
    for study in STUDIES:
        rows = [(family, arms[study]) for family, arms in families.items() if study in arms]
        if not rows:
            continue
        title = arm_title(rows[0][0], study)[0]
        lines += [
            f"### Cross-model summary -- {title}",
            "",
            "| Model | Axis | B.Acc by level | Best | ρ B.Acc | Page p ↑ / ↓ (B.Acc) | "
            "ρ F1 | Page p ↑ / ↓ (F1) |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for family, arm in rows:
            _title, axis, _label = arm_title(family, study)
            best = _best(arm)
            acc, f1 = arm["trend"]["accuracy"], arm["trend"]["f1_score"]
            incomplete = "" if arm["complete"] else " (incomplete)"
            lines.append(
                f"| {FAMILIES[family].label}{incomplete} | {axis} | {_level_means(arm)} | "
                f"{'--' if best is None else '`' + best['variant'] + '`'} | "
                f"{_stat(acc.get('spearman'), 'rho', '+.3f')} | "
                f"{_p(_page_p(acc, 'increasing'))} / {_p(_page_p(acc, 'decreasing'))} | "
                f"{_stat(f1.get('spearman'), 'rho', '+.3f')} | "
                f"{_p(_page_p(f1, 'increasing'))} / {_p(_page_p(f1, 'decreasing'))} |"
            )
        lines.append("")
    if lines:
        lines += [
            "B.Acc by level is the mean over all per-fold scores, in %. ρ is Spearman's "
            "against the arm's level; Page p ↑ tests a trend increasing with the level, ↓ a "
            "decreasing one. Each row is a separate within-model test.",
            "",
        ]
    return lines


def render_arm_markdown(family: str, study: str, arm: OrderedDict) -> List[str]:
    title, axis, level_label = arm_title(family, study)
    lines = [f"#### {title}", ""]
    if not arm["complete"]:
        lines += [
            f"> Incomplete: only {len(arm['variants'])} of the arm's variants have "
            f"results (missing: {', '.join('`' + key + '`' for key in arm['missing'])}). "
            "Tests below use what is present.",
            "",
        ]
    header = (
        f"| Variant | {level_label} | Params | RF (samples) | "
        "B.Acc | Macro F1 | B.Acc (std over rounds) |"
    )
    lines += [header, "|---|---|---|---|---|---|---|"]
    for row in arm["variants"]:
        acc, f1 = row["metrics"]["accuracy"], row["metrics"]["f1_score"]
        lines.append(
            f"| `{row['variant']}` | {int(row['level'])} | "
            f"{row['num_parameters']:,} | {row['receptive_field_samples']} | "
            f"{_pm(acc)} | {_pm(f1)} | {_pm(acc, 'std_over_rounds')} |"
        )
    lines.append("")

    for metric, label in METRICS:
        lines += [
            f"**{label} -- adjacent paired Wilcoxon** (n pairs = folds shared by both "
            "variants; Holm-corrected within this arm)",
            "",
            "| Comparison | Δ mean (pp) | Δ median (pp) | n | p | p (Holm) | sig. |",
            "|---|---|---|---|---|---|---|",
        ]
        for row in arm["comparisons"][metric]:
            sig = row.get("significant_holm_0.05")
            verdict = "--" if sig is None else ("yes" if sig else "no")
            lines.append(
                f"| `{row['baseline']}` → `{row['candidate']}` | "
                f"{_delta_pp(row.get('mean_difference'))} | "
                f"{_delta_pp(row.get('median_difference'))} | "
                f"{row.get('n_pairs', '--')} | {_p(row.get('p_value'))} | "
                f"{_p(row.get('p_value_holm'))} | {verdict} |"
            )
        trend = arm["trend"][metric]
        up = trend.get("page_increasing")
        down = trend.get("page_decreasing")
        spearman = trend.get("spearman", {})
        lines += [
            "",
            f"**{label} -- trend across {axis}** "
            f"(Page's L over {trend.get('n_blocks', 0)} folds x "
            f"{trend.get('n_variants', 0)} variants)",
            "",
            f"- Page's L, increasing with {axis}: "
            f"L = {_stat(up, 'statistic', '.1f')}, "
            f"p = {_p(up.get('p_value') if up else None)}",
            f"- Page's L, decreasing with {axis}: "
            f"L = {_stat(down, 'statistic', '.1f')}, "
            f"p = {_p(down.get('p_value') if down else None)}",
            f"- Spearman ρ vs {axis}: {_stat(spearman, 'rho', '+.3f')}, "
            f"p = {_p(spearman.get('p_value'))}, n = {spearman.get('n', '--')}",
            "",
        ]
    return lines


def render_markdown(report: OrderedDict) -> str:
    lines: List[str] = [
        "# Ablations -- depth (R2.2) and receptive field (R2.3)",
        "",
        "Values are percentages, `mean ± std` over **all per-fold scores** (the convention "
        "Table 4's caption states). `std over rounds` is the alternative convention stored "
        "in `results.summary` of each JSON, shown for transparency.",
        "",
        "`Balanced Accuracy` is `balanced_accuracy_score`; `Macro F1` is `f1_score(average='macro')`.",
        "",
        "Depth is counted in each model's own unit: conv blocks for the 1D-CNN, conv layers "
        "for AlexNet, weighted layers for ResNet (10/18/26/34).",
        "",
    ]
    for dataset, families in report.items():
        lines += [f"## {dataset}", ""]
        lines += render_summary_markdown(families)
        for family, studies in families.items():
            lines += [f"### {FAMILIES[family].label}", ""]
            for study, arm in studies.items():
                lines += render_arm_markdown(family, study, arm)
    return "\n".join(lines).rstrip() + "\n"


def _latex_pm(stats: OrderedDict, dispersion: str = "std_over_folds") -> str:
    return _pm(stats, dispersion).replace("±", "\\pm")


def _latex_p(value: Optional[float]) -> str:
    return _p(value).replace("<", "$<$")


def render_latex(report: OrderedDict) -> str:
    blocks: List[str] = [
        "% Generated by run_experiments_ablation.py --report. Values are percentages,",
        "% mean +/- std over all per-fold scores of the unbiased multiround protocol.",
        "",
    ]
    for dataset, families in report.items():
        for study in STUDIES:
            rows = [(family, arms[study]) for family, arms in families.items() if study in arms]
            if not rows:
                continue
            title = arm_title(rows[0][0], study)[0]
            width = max(len(arm["variants"]) for _family, arm in rows)
            blocks += [
                "\\begin{table}[ht]",
                "\\centering",
                f"\\caption{{{title} on {dataset}, across models. Balanced Accuracy (\\%) of "
                "each variant, mean across all cross-validation folds, in increasing order "
                "of the arm's level; $\\rho$ is Spearman's correlation with the level and "
                "$p_\\uparrow$, $p_\\downarrow$ are Page's trend test for an increasing and "
                "a decreasing trend.}",
                f"\\label{{tab:ablation_summary_{study}_{dataset.lower()}}}",
                f"\\begin{{tabular}}{{@{{}}l l {'c ' * width}r r r@{{}}}}",
                "\\toprule",
                "\\textbf{Model} & \\textbf{Levels} & "
                f"\\multicolumn{{{width}}}{{c}}{{\\textbf{{B.Acc by level}}}} & "
                "$\\rho$ & $p_\\uparrow$ & $p_\\downarrow$ \\\\ \\midrule",
            ]
            for family, arm in rows:
                levels = "/".join(str(int(row["level"])) for row in arm["variants"])
                cells = [_pct(row["metrics"]["accuracy"]["mean"]) for row in arm["variants"]]
                cells += ["--"] * (width - len(cells))
                trend = arm["trend"]["accuracy"]
                rho = _stat(trend.get("spearman"), "rho", "+.3f")
                blocks.append(
                    f"{FAMILIES[family].label} & {levels} & {' & '.join(cells)} & "
                    f"{rho if rho == '--' else '$' + rho + '$'} & "
                    f"{_latex_p(_page_p(trend, 'increasing'))} & "
                    f"{_latex_p(_page_p(trend, 'decreasing'))} \\\\"
                )
            blocks += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]

        for family, studies in families.items():
            for study, arm in studies.items():
                title, _axis, level_label = arm_title(family, study)
                label = f"tab:ablation_{family}_{study}_{dataset.lower()}"
                blocks += [
                    "\\begin{table}[ht]",
                    "\\centering",
                    f"\\caption{{{FAMILIES[family].label} {title} on {dataset}. Balanced "
                    "Accuracy (B.Acc) and Macro F1-Score (\\%), mean $\\pm$ standard deviation "
                    "across all cross-validation folds. RF is the theoretical receptive field "
                    "of the convolutional stack, in input samples.}",
                    f"\\label{{{label}}}",
                    "\\begin{tabular}{@{}l c r r c c@{}}",
                    "\\toprule",
                    f"\\textbf{{Variant}} & \\textbf{{{level_label}}} & \\textbf{{Params}} & "
                    "\\textbf{RF} & \\textbf{B.Acc} & \\textbf{Macro F1} \\\\ \\midrule",
                ]
                for row in arm["variants"]:
                    acc, f1 = row["metrics"]["accuracy"], row["metrics"]["f1_score"]
                    name = row["variant"].replace("_", "-")
                    blocks.append(
                        f"{name} & {int(row['level'])} & {row['num_parameters']:,} & "
                        f"{row['receptive_field_samples']} & "
                        f"${_latex_pm(acc)}$ & ${_latex_pm(f1)}$ \\\\"
                    )
                blocks += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(blocks)


def build(output_dir: str) -> Tuple[OrderedDict, str, str]:
    """Load, analyse and render. Returns ``(report, markdown, latex)``."""
    report = analyse(load_results(output_dir))
    return report, render_markdown(report), render_latex(report)
