"""Controlled ablations answering Reviewer #2 (``review/reviwers_comments.md``).

``run_experiments_ablation.py`` at the repository root is the entry point; everything it
needs beyond the shared benchmark code in ``src/`` lives here:

===============  ==============================================================
`models`         parametric CNN1D: depth and initial kernel size as arguments
`registry`       the 21 ablation experiments and the JSON document they produce
`report`         ablation results -> tables and significance tests
===============  ==============================================================

The training loop (``src.experiment``), the datasets (``src.datasets``), the unbiased folds
(``src.folds``) and the result serialisation (``src.serialize``) are reused from the main
benchmark unchanged, so the ablation numbers stay comparable with the published ones.
"""

__all__ = ["models", "registry", "report"]
