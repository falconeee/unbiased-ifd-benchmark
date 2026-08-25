"""Supporting code for the unbiased IFD benchmark.

``run_experiments.py`` at the repository root is the entry point; everything it needs lives
here:

===============  ==============================================================
`registry`       the 108 experiments and their hyperparameters
`datasets`       the 6 datasets: download, transform, grouping, resampling
`folds`          unbiased folds, single and multi round
`models`         the 9 architectures
`experiment`     ``DeepLearningExperiment``: cross validation and training loop
`serialize`      results -> the JSON schema written under ``results/``
===============  ==============================================================
"""

import matplotlib as _matplotlib

# Set before anything imports pyplot (``signalAI.utils.experiment_result`` does): the
# scripts run headless, where the default macOS/Qt backends fail or open windows.
_matplotlib.use("Agg")

__all__ = ["datasets", "experiment", "folds", "models", "registry", "serialize"]
