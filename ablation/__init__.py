"""Controlled ablations answering Reviewer #2 (``review/reviwers_comments.md``).

``run_experiments_ablation.py`` at the repository root is the entry point; everything it
needs beyond the shared benchmark code in ``src/`` lives here:

===============  ==============================================================
`models`         parametric 1D-CNN, AlexNet and ResNet-18: depth and first
                 kernel as arguments
`registry`       the 63 ablation experiments (3 models x 7 variants x 3
                 datasets) and the JSON document they produce
`report`         ablation results -> per-model and cross-model tables and
                 significance tests
===============  ==============================================================

The training loop (``src.experiment``), the datasets (``src.datasets``), the unbiased folds
(``src.folds``), the result serialisation (``src.serialize``) and each model's published
hyperparameters (``src.registry``) are reused from the main benchmark unchanged, so the
ablation numbers stay comparable with the published ones.

Typical use -- only the depth arm of AlexNet and ResNet-18, leaving the 1D-CNN alone::

    python run_experiments_ablation.py --model alexnet --model resnet18 --study depth --list
    python run_experiments_ablation.py --model alexnet --model resnet18 --study depth \\
        --resume --keep-going
"""

__all__ = ["models", "registry", "report"]
