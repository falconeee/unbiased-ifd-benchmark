"""Runner package for the unbiased IFD benchmark.

Executes the experiments defined in the notebooks under ``single_round_experiments/`` and
``multi_round_experiments/`` from a plain Python script, and writes one JSON per experiment
using the same schema as ``output/``.
"""

import matplotlib as _matplotlib

# Set before anything imports pyplot (``signalAI.utils.experiment_result`` does): the
# scripts run headless, where the default macOS/Qt backends fail or open windows.
_matplotlib.use("Agg")

__all__ = ["datasets", "experiment", "folds", "models", "registry", "serialize"]
