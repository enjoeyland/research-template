"""Small MLP classifier -- a placeholder showing the LightningModule conventions of this repo.

Conventions worth keeping in your own modules:
  * metrics live in a ``TaskMetrics`` (src/metrics, configured in ``configs/metrics/*.yaml`` and injected
    through the model config); it also provides ``val/<name>_best`` and ``val/overfit_gap``,
  * ``on_step`` returns metric OBJECTS -> log them with ``log_dict(..., on_epoch=True)`` (Lightning resets
    them each epoch -- never call ``.reset()`` by hand, CLAUDE.md §3),
  * ``self.metrics.reset_best()`` once from ``on_train_start``,
  * the monitored metric (``model.metrics.monitor_metric``) must be one the module logs.
"""

from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule

from src.metrics import TaskMetrics

Batch = Tuple[torch.Tensor, torch.Tensor]


class ToyModule(LightningModule):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        n_features: int,
        n_classes: int,
        metrics: TaskMetrics,
        hidden_size: int = 32,
    ) -> None:
        super().__init__()
        # the metrics object is a module of its own, not a hyperparameter
        self.save_hyperparameters(logger=False, ignore=["metrics"])
        self.metrics = metrics

        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_features, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, n_classes),
        )
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def model_step(self, batch: Batch):
        x, y = batch
        logits = self.forward(x)
        return self.criterion(logits, y), torch.argmax(logits, dim=1), y

    def on_train_start(self) -> None:
        # drops what the pre-training sanity-check validation pass wrote into the best-score tracker
        self.metrics.reset_best()

    def _step(self, split: str, batch: Batch, batch_idx: int) -> torch.Tensor:
        loss, preds, target = self.model_step(batch)
        key = "val" if split == "valid" else split
        logged = {f"{key}/loss": loss}
        logged.update(self.metrics.on_step(split, None, batch, batch_idx, preds=preds, target=target))
        self.log_dict(logged, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def _epoch_end(self, split: str) -> None:
        values = self.metrics.on_epoch_end(split)
        if values:
            self.log_dict(values, prog_bar=True, sync_dist=True)

    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        return self._step("train", batch, batch_idx)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")  # also caches the train score that val/overfit_gap compares against

    def validation_step(self, batch: Batch, batch_idx: int) -> None:
        self._step("valid", batch, batch_idx)

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("valid")  # logs val/acc_best and val/overfit_gap too

    def test_step(self, batch: Batch, batch_idx: int) -> None:
        self._step("test", batch, batch_idx)

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self) -> Dict[str, Any]:
        return {"optimizer": self.hparams.optimizer(params=self.trainer.model.parameters())}
