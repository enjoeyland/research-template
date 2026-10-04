"""Regression tests for src/metrics.

IMPORTANT: TaskMetrics/MetricGroup have no reset() calls of their own in on_epoch_end -- the per-epoch
(and sanity-check) reset is Lightning's own doing. It only kicks in when a metric object is actually
routed through `self.log()`/`log_dict(..., on_epoch=True)`, which is exactly what `on_step` returning
`self` (not the computed value) is for. So these tests must run through a real `lightning.Trainer` --
calling on_step()/on_epoch_end() directly in plain Python has no such auto-reset and would give a
false negative (CLAUDE.md §3).
"""

import torch
from lightning import LightningModule, Trainer
from torch.utils.data import DataLoader, Dataset

from src.metrics import MetricGroup, MulticlassAccuracy, TaskMetrics


class _AccGroup(MetricGroup):
    def init_metrics(self) -> None:
        for split in (self.metrics_train, self.metrics_valid, self.metrics_test):
            split["acc"] = MulticlassAccuracy(num_classes=2)


class _FixedDataset(Dataset):
    """Every item is the same fixed (pred, target) pair."""

    def __init__(self, pred: int, target: int, n: int = 4) -> None:
        self.pred, self.target, self.n = pred, target, n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        return torch.tensor(self.pred), torch.tensor(self.target)


class _Module(LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.metrics = TaskMetrics([_AccGroup()], monitor_metric="val/acc", monitor_mode="max")
        self.dummy = torch.nn.Linear(1, 1)

    def training_step(self, batch, batch_idx):
        pred, target = batch
        self.log_dict(
            self.metrics.on_step("train", None, batch, batch_idx, preds=pred, target=target),
            on_step=False,
            on_epoch=True,
        )
        return (self.dummy(pred.float().unsqueeze(-1)) * 0).sum()  # zero grad, just needs a graph

    def on_train_epoch_end(self) -> None:
        self.last_train_epoch_metrics = self.metrics.on_epoch_end("train")

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


def _fit_one_epoch(model: _Module, pred: int, target: int) -> float:
    trainer = Trainer(
        max_epochs=1,
        accelerator="cpu",
        enable_progress_bar=False,
        enable_checkpointing=False,
        logger=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, DataLoader(_FixedDataset(pred, target), batch_size=1))
    return float(model.last_train_epoch_metrics["train/acc"])


def test_metric_accuracy_is_isolated_per_epoch() -> None:
    """A perfect epoch followed by a completely wrong epoch (same model instance, no manual reset
    anywhere) must report 1.0 then 0.0 -- not a blended 0.5."""
    model = _Module()
    assert _fit_one_epoch(model, pred=1, target=1) == 1.0
    assert _fit_one_epoch(model, pred=1, target=0) == 0.0


def test_metric_on_step_returns_the_metric_object() -> None:
    """on_step must return `self` (not the computed value) -- that is what lets Lightning's own
    log()-based mechanism auto-compute/reset it at each epoch boundary."""
    metric = MulticlassAccuracy(num_classes=2)
    result = metric.on_step("valid", None, None, 0, preds=torch.tensor([1, 0]), target=torch.tensor([1, 0]))
    assert result is metric


class _FixedScore(TaskMetrics):
    """TaskMetrics whose monitored score is injected, to test the derived trackers in isolation."""

    score = None

    def compute_monitor_score(self, metrics):
        return self.score


def _valid_epoch(mode: str, train: float, val: float):
    metrics = _FixedScore([_AccGroup()], monitor_metric="val/acc", monitor_mode=mode)
    metrics.reset_best()
    metrics._last_train_score = train
    metrics.score = val
    return metrics.on_epoch_end("valid")


def test_overfit_gap_positive_means_overfitting() -> None:
    """Positive gap = overfitting for BOTH directions: `max` -> train - val, `min` -> val - train."""
    out = _valid_epoch("max", train=0.9, val=0.6)  # accuracy: train above val
    assert abs(out["val/overfit_gap"] - 0.3) < 1e-6 and abs(out["val/acc_best"] - 0.6) < 1e-6
    out = _valid_epoch("min", train=0.2, val=0.5)  # loss: val above train
    assert abs(out["val/overfit_gap"] - 0.3) < 1e-6 and abs(out["val/acc_best"] - 0.5) < 1e-6
