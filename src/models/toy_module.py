"""Small MLP classifier -- a placeholder showing the LightningModule conventions of this repo.

Conventions worth keeping in your own modules (details: CLAUDE.md §3):
  * ``forward`` returns ONE ``ModelOutput`` per step; the loss and the metrics both read it, so nothing is
    computed twice. Put only what the model alone can produce into it (logits, sampled masks, features);
    softmax-style preprocessing belongs to the loss / metric,
  * the loss is a ``CompositeLoss`` of named terms (src/losses, configs/losses/*.yaml) injected through the model
    config; it returns ``{"loss": total, "loss_<term>": ...}`` -- backprop ``loss``, log everything,
  * metrics live in a ``TaskMetrics`` (src/metrics, configs/metrics/*.yaml); ``on_step`` returns metric OBJECTS
    -> ``log_dict(..., on_epoch=True)`` (Lightning resets them each epoch -- never call ``.reset()`` by hand),
  * ``val/<name>_best`` and ``val/overfit_gap`` need no code here: the ``MetricTrends`` callback derives them,
  * the monitored metric (``model.metrics.monitor_metric``) must be one the module logs.
"""

from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule

from src.losses import CompositeLoss
from src.metrics import TaskMetrics
from src.utils.model_output import ModelOutput

Batch = Tuple[torch.Tensor, torch.Tensor]


class ToyModule(LightningModule):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        n_features: int,
        n_classes: int,
        metrics: TaskMetrics,
        loss: CompositeLoss,
        hidden_size: int = 32,
    ) -> None:
        super().__init__()
        # metrics / loss are modules of their own, not hyperparameters
        self.save_hyperparameters(logger=False, ignore=["metrics", "loss"])
        self.metrics = metrics
        self.loss_fn = loss

        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_features, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, n_classes),
        )

    def forward(self, batch: Batch) -> ModelOutput:
        x, y = batch
        return ModelOutput(logits=self.net(x), target=y)

    def _step(self, split: str, batch: Batch, batch_idx: int) -> torch.Tensor:
        outputs = self.forward(batch)
        loss_dict = self.loss_fn(outputs, batch)
        key = "val" if split == "valid" else split
        logged = {f"{key}/{name}": value for name, value in loss_dict.items()}
        logged.update(self.metrics.on_step(split, outputs, batch, batch_idx))
        self.log_dict(logged, on_step=False, on_epoch=True, prog_bar=True)
        return loss_dict["loss"]

    def _epoch_end(self, split: str) -> None:
        values = self.metrics.on_epoch_end(split)
        if values:
            self.log_dict(values, prog_bar=True, sync_dist=True)

    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        return self._step("train", batch, batch_idx)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def validation_step(self, batch: Batch, batch_idx: int) -> None:
        self._step("valid", batch, batch_idx)

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("valid")

    def test_step(self, batch: Batch, batch_idx: int) -> None:
        self._step("test", batch, batch_idx)

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self) -> Dict[str, Any]:
        return {"optimizer": self.hparams.optimizer(params=self.trainer.model.parameters())}
