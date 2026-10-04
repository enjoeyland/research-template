"""Small MLP classifier -- a placeholder showing the LightningModule conventions of this repo.

Conventions worth keeping in your own modules:
  * log ``val/<metric>`` and ``val/<metric>_best`` so ``model.metrics.monitor_metric`` can point at it,
  * log metric *objects* with ``on_epoch=True`` (Lightning resets them) -- never call ``.reset()`` by hand
    (see CLAUDE.md §3).
"""

from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy


class ToyModule(LightningModule):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        n_features: int,
        n_classes: int,
        hidden_size: int = 32,
        metrics: Dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        # `metrics` only carries monitor_metric / monitor_mode for the callbacks, not a module hparam
        self.save_hyperparameters(logger=False, ignore=["metrics"])

        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_features, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, n_classes),
        )
        self.criterion = torch.nn.CrossEntropyLoss()

        task = "multiclass"
        self.train_acc = Accuracy(task=task, num_classes=n_classes)
        self.val_acc = Accuracy(task=task, num_classes=n_classes)
        self.test_acc = Accuracy(task=task, num_classes=n_classes)
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def model_step(self, batch: Tuple[torch.Tensor, torch.Tensor]):
        x, y = batch
        logits = self.forward(x)
        return self.criterion(logits, y), torch.argmax(logits, dim=1), y

    def on_train_start(self) -> None:
        # sanity-check validation batches would otherwise leak into the best value
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        loss, preds, targets = self.model_step(batch)
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        loss, preds, targets = self.model_step(batch)
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        self.val_acc_best(self.val_acc.compute())
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch, batch_idx: int) -> None:
        loss, preds, targets = self.model_step(batch)
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> Dict[str, Any]:
        return {"optimizer": self.hparams.optimizer(params=self.trainer.model.parameters())}
