"""``DeepLearningExperiment``, vendored from the notebook cells.

The notebooks define this class inline (it is not part of ``signalAI==0.0.8``, and the
version on the SignAI-Framework ``main`` branch has different training semantics -- it has
no autoencoder pre-training phase). It is kept here so the script trains exactly like the
notebooks did.

Every deviation from the notebook source is marked with ``# FIX:`` and explained in
``README.md``.
"""

from __future__ import annotations

import copy
import os
import random
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")  # FIX: notebooks rely on an interactive backend; scripts run headless.

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset, random_split

from signalAI.utils.experiment_result import ExperimentResults, FoldResults
from signalAI.utils.metrics import calculate_metrics


class TorchVibrationDataset(Dataset):
    """Wrapper to convert dataset samples into Torch tensors."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def kl_divergence(rho, rho_hat):
    rho_hat = torch.mean(rho_hat, dim=0)
    rho = torch.tensor([rho] * len(rho_hat), device=rho_hat.device)
    epsilon = 1e-7
    term1 = rho * torch.log((rho + epsilon) / (rho_hat + epsilon))
    term2 = (1 - rho) * torch.log((1 - rho + epsilon) / (1 - rho_hat + epsilon))
    return torch.sum(term1 + term2)


def seed_everything(seed: int) -> None:
    """FIX: the notebooks never seed, so the validation split changed on every run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _reshape_for(model: nn.Module, xb: torch.Tensor) -> torch.Tensor:
    """The shape juggling the notebook repeats before every forward pass."""
    if xb.ndim == 2 and any(isinstance(m, nn.Conv1d) for m in model.modules()):
        return xb.unsqueeze(1)
    if xb.ndim == 2 and any(isinstance(m, nn.Conv2d) for m in model.modules()):
        side = int(np.sqrt(xb.shape[1]))
        return xb.view(xb.size(0), 1, side, side)
    return xb


class DeepLearningExperiment:
    """Nested cross validation over pre-assigned folds, with optional AE pre-training."""

    def __init__(
        self,
        name: str,
        description: str,
        dataset=None,
        data_fold_idxs: np.ndarray = None,
        model: nn.Module = None,
        criterion: Optional[nn.Module] = None,
        reconstruction_criterion: Optional[nn.Module] = None,
        recon_loss_weight: float = 1.0,
        sparsity_target: Optional[float] = None,
        sparsity_weight: float = 0.0,
        pretrain_epochs: int = 0,
        optimizer_class=optim.Adam,
        batch_size: int = 32,
        lr: float = 1e-3,
        num_epochs: int = 20,
        val_split: float = 0.2,
        output_dir: str = "results_torch",
        device: str = None,
        # --- additions (all default to the notebook behaviour) ---
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        seed: int = 42,
        save_artifacts: bool = True,
        progress_every: int = 5,
    ):
        self.name = name
        self.description = description
        self.dataset = dataset
        self.model = model
        self.data_fold_idxs = np.asarray(data_fold_idxs)
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.pretrain_epochs = pretrain_epochs
        self.val_split = val_split
        self.optimizer_class = optimizer_class
        self.lr = lr
        self.seed = seed
        self.save_artifacts = save_artifacts
        self.progress_every = progress_every
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.reconstruction_criterion = reconstruction_criterion
        self.recon_loss_weight = recon_loss_weight
        self.sparsity_target = sparsity_target
        self.sparsity_weight = sparsity_weight

        self.is_sae_task = self.sparsity_target is not None and self.sparsity_weight > 0.0
        self.is_autoencoder_task = (
            reconstruction_criterion is not None
            or self.is_sae_task
            or "AE1D" in model.__class__.__name__
        )
        if self.is_sae_task and self.reconstruction_criterion is None:
            print("Warning: SAE task detected but no reconstruction_criterion. Defaulting to MSELoss.")
            self.reconstruction_criterion = nn.MSELoss()

        # FIX: the notebook wrapped the model in ``nn.DataParallel`` *after*
        # ``super().__init__`` had already stored the unwrapped one, so ``_train_one_fold``
        # always deep-copied the unwrapped model and the branch never took effect. The
        # prototype is kept on CPU here; only each fold's copy is moved to the device.
        self.original_model = model

        self.n_outer_folds = len(np.unique(self.data_fold_idxs))
        if self.save_artifacts:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Recorded for the result files; the notebooks only printed every 5th epoch.
        self.history: Dict[int, Dict[str, List[dict]]] = {}
        self.fold_errors: List[dict] = []

        self.prepare_data(X=X, y=y)

    # ------------------------------------------------------------------ data
    def prepare_data(self, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None):
        """FIX: accept pre-loaded arrays.

        The notebook re-reads the whole ``DeepDataset`` in ``__init__``; inside a multiround
        loop that means loading every sample from disk once per round (8x per model, 72x per
        dataset). ``load_xy`` below does it once and the arrays are handed in.
        """
        if X is not None and y is not None:
            self.X, self.y = X, y
            self.label_encoder = None
            return
        self.X, self.y, self.label_encoder = load_xy(self.dataset)

    # ------------------------------------------------------------------ train
    def _train_one_fold(self, X_train, y_train, X_test, y_test, fold_idx: int) -> FoldResults:
        train_dataset = TorchVibrationDataset(X_train, y_train)
        test_dataset = TorchVibrationDataset(X_test, y_test)

        val_size = int(self.val_split * len(train_dataset))
        # FIX: an empty validation split divided by zero when computing the epoch loss.
        if val_size < 1:
            if len(train_dataset) < 2:
                raise ValueError(
                    f"fold {fold_idx}: only {len(train_dataset)} training samples, "
                    f"cannot build a validation split"
                )
            val_size = 1
        train_size = len(train_dataset) - val_size
        # FIX: seeded generator, so the split is reproducible across runs.
        split_generator = torch.Generator().manual_seed(self.seed + fold_idx)
        train_dataset, val_dataset = random_split(
            train_dataset, [train_size, val_size], generator=split_generator
        )

        # FIX: a trailing batch of exactly one sample makes BatchNorm1d raise
        # "Expected more than 1 value per channel when training". In the notebook that
        # exception was swallowed by ``run()`` and the whole fold silently vanished from the
        # averages. Dropping it only kicks in for that one pathological size.
        drop_last = len(train_dataset) % self.batch_size == 1
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=drop_last
        )
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        model = copy.deepcopy(self.original_model).to(self.device)
        model_core = model

        has_ae_structure = (
            hasattr(model_core, "encoder")
            and hasattr(model_core, "decoder")
            and hasattr(model_core, "classifier")
        )

        pretrain_history: List[dict] = []
        supervised_history: List[dict] = []

        # ---------------------------------------------------- autoencoder pre-training
        if self.is_autoencoder_task and self.pretrain_epochs > 0 and has_ae_structure:
            print(f"[Fold {fold_idx}] AutoEncoder training ({self.pretrain_epochs} epochs)...")
            optimizer_ae = self.optimizer_class(
                [
                    {"params": model_core.encoder.parameters()},
                    {"params": model_core.decoder.parameters()},
                ],
                lr=self.lr,
            )

            for epoch in range(self.pretrain_epochs):
                epoch_start = time.time()
                model.train()
                running_recon_loss = 0.0
                seen = 0

                for xb, _ in train_loader:
                    xb = xb.to(self.device)
                    input_data = xb
                    xb = _reshape_for(model, xb)

                    optimizer_ae.zero_grad()
                    outputs = model(xb)
                    if not isinstance(outputs, tuple):
                        continue

                    reconstruction = outputs[1]
                    loss = self.reconstruction_criterion(reconstruction, input_data)
                    if self.is_sae_task and len(outputs) > 2:
                        latent_features = outputs[2]
                        loss = loss + self.sparsity_weight * kl_divergence(
                            self.sparsity_target, latent_features
                        )

                    loss.backward()
                    optimizer_ae.step()
                    running_recon_loss += loss.item() * input_data.size(0)
                    seen += input_data.size(0)

                avg_recon_loss = running_recon_loss / max(seen, 1)
                pretrain_history.append(
                    {
                        "epoch": epoch + 1,
                        "total_epochs": self.pretrain_epochs,
                        "recon_loss": avg_recon_loss,
                        "time_seconds": time.time() - epoch_start,
                    }
                )
                if (epoch + 1) % self.progress_every == 0 or epoch == 0:
                    print(
                        f"  [Pre-train] Epoch {epoch+1}/{self.pretrain_epochs} "
                        f"Recon Loss: {avg_recon_loss:.4f}"
                    )

        # ------------------------------------------------------------ classifier phase
        print(f"[Fold {fold_idx}] Classifier training ({self.num_epochs} epochs)...")
        if has_ae_structure and self.is_autoencoder_task:
            optimizer_clf = self.optimizer_class(
                [
                    {"params": model_core.encoder.parameters()},
                    {"params": model_core.classifier.parameters()},
                ],
                lr=self.lr,
            )
        else:
            optimizer_clf = self.optimizer_class(model.parameters(), lr=self.lr)

        train_losses, val_losses = [], []

        for epoch in range(self.num_epochs):
            epoch_start = time.time()
            model.train()
            running_loss = 0.0
            seen = 0

            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                xb = _reshape_for(model, xb)

                optimizer_clf.zero_grad()
                outputs = model(xb)
                classification_output = outputs[0] if isinstance(outputs, tuple) else outputs

                loss = self.criterion(classification_output, yb)
                loss.backward()
                optimizer_clf.step()
                running_loss += loss.item() * xb.size(0)
                seen += xb.size(0)

            avg_train_loss = running_loss / max(seen, 1)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    xb = _reshape_for(model, xb)
                    outputs = model(xb)
                    classification_output = outputs[0] if isinstance(outputs, tuple) else outputs
                    val_loss += self.criterion(classification_output, yb).item() * xb.size(0)

            avg_val_loss = val_loss / len(val_loader.dataset)

            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            epoch_time = time.time() - epoch_start
            supervised_history.append(
                {
                    "epoch": epoch + 1,
                    "total_epochs": self.num_epochs,
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "time_seconds": epoch_time,
                }
            )
            if (epoch + 1) % self.progress_every == 0 or epoch == 0:
                print(
                    f"  [Supervised] Epoch {epoch+1}/{self.num_epochs} "
                    f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
                    f"Time: {epoch_time:.2f}s"
                )

        self.history[fold_idx] = {"pretrain": pretrain_history, "supervised": supervised_history}

        if self.save_artifacts:
            fig = plt.figure()
            plt.plot(train_losses, label="Train Loss (Clf)")
            plt.plot(val_losses, label="Val Loss (Clf)")
            plt.legend()
            plt.title(f"Loss Curve - Fold {fold_idx}")
            fig.savefig(os.path.join(self.dir_path, f"loss_curve_fold{fold_idx}.png"))
            plt.close(fig)  # FIX: the notebook leaked one figure per fold.
            torch.save(model_core.state_dict(), os.path.join(self.dir_path, f"model_fold{fold_idx}.pt"))

        # ------------------------------------------------------------------------ test
        y_true, y_pred, y_proba = [], [], []
        model.eval()
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                xb = _reshape_for(model, xb)
                outputs = model(xb)
                classification_output = outputs[0] if isinstance(outputs, tuple) else outputs
                probs = torch.softmax(classification_output, dim=1)
                preds = torch.argmax(probs, dim=1)
                y_true.extend(yb.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
                y_proba.extend(probs.cpu().numpy())

        metrics = calculate_metrics(np.array(y_true), np.array(y_pred), np.array(y_proba))
        return FoldResults(fold_idx, np.array(y_true), np.array(y_pred), np.array(y_proba), metrics)

    # -------------------------------------------------------------------------- run
    def run(self) -> ExperimentResults:
        seed_everything(self.seed)
        self.start_time = time.strftime("%Y%m%d_%H%M%S")
        self.dir_path = os.path.join(self.output_dir, f"results_{self.name}_{self.start_time}")
        if self.save_artifacts:
            os.makedirs(self.dir_path, exist_ok=True)

        results = ExperimentResults(
            experiment_name=self.name,
            description=self.description,
            model_name=self.original_model.__class__.__name__,
            feature_names=None,
            config={
                "n_outer_folds": self.n_outer_folds,
                "pretrain_epochs": self.pretrain_epochs,
                "finetune_epochs": self.num_epochs,
                "batch_size": self.batch_size,
                "lr": self.lr,
            },
        )

        for outer_fold in range(self.n_outer_folds):
            print(f"\n=== Outer Fold {outer_fold+1}/{self.n_outer_folds} ===")
            train_mask = self.data_fold_idxs != outer_fold
            test_mask = self.data_fold_idxs == outer_fold

            try:
                fold_result = self._train_one_fold(
                    self.X[train_mask], self.y[train_mask],
                    self.X[test_mask], self.y[test_mask],
                    outer_fold,
                )
                results.add_fold_result(fold_result)
                print(
                    f"  Result: Acc={fold_result.metrics['accuracy']:.4f}, "
                    f"F1={fold_result.metrics['f1']:.4f}"
                )
            except Exception as exc:  # noqa: BLE001 -- keep the notebook's resilience
                print(f"Error in fold {outer_fold}: {exc}")
                traceback.print_exc()
                # FIX: the notebook only printed this, so a failed fold disappeared from the
                # averages with no trace in the saved results.
                self.fold_errors.append(
                    {
                        "fold": outer_fold,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }
                )

        if not results.folds:
            raise RuntimeError(f"experiment {self.name}: every fold failed")

        results.calculate_overall_metrics()
        print("\n=== Final Results ===")
        print(f"Mean Accuracy: {results.overall_metrics['accuracy']:.4f}")
        return results


def load_xy(deep_dataset):
    """Flatten a ``DeepDataset`` into ``(X, y, label_encoder)``.

    FIX: ``np.array(features)`` keeps float64 in the notebook (242 MB for CWRU12k alone);
    the tensors are built as float32 anyway, so the cast happens up front.
    """
    features, labels = [], []
    for sample in deep_dataset:
        features.append(sample["signal"][0])
        labels.append(sample["metainfo"]["label"])
    X = np.asarray(features, dtype=np.float32)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    return X, y, label_encoder
