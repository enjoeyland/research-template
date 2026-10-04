"""MetricTrends (val/<name>_best, val/overfit_gap) through a real Trainer: the pairing of train/val epochs and the
sanity-check isolation only exist inside Lightning's loop order."""

import torch
from lightning import Callback, LightningModule, Trainer
from torch.utils.data import DataLoader, TensorDataset

from src.utils.callbacks import MetricTrends

TRAIN = [0.60, 0.90, 0.95, 0.99]
VAL = [0.50, 0.70, 0.60, 0.80]


class _Scripted(LightningModule):
    """Logs scripted train/val values per epoch (val is 99 during the sanity check to catch pollution)."""

    def __init__(self) -> None:
        super().__init__()
        self.dummy = torch.nn.Linear(1, 1)

    def training_step(self, batch, batch_idx):
        self.log("train/acc", TRAIN[self.current_epoch], on_epoch=True)
        return self.dummy(batch[0]).sum() * 0

    def validation_step(self, batch, batch_idx):
        value = 99.0 if self.trainer.sanity_checking else VAL[self.current_epoch]
        self.log("val/acc", value, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


class _Record(Callback):
    """Runs after MetricTrends (callback order) and records what it logged."""

    def __init__(self) -> None:
        self.rows = []

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        m = trainer.callback_metrics
        self.rows.append({k: float(m[k]) for k in ("val/acc_best", "val/overfit_gap") if k in m})


def _loader():
    return DataLoader(TensorDataset(torch.zeros(4, 1)), batch_size=2)


def _fit(mode: str):
    trends, rec = MetricTrends("val/acc", mode), _Record()
    trainer = Trainer(
        max_epochs=len(VAL), accelerator="cpu", logger=False, enable_checkpointing=False,
        enable_progress_bar=False, callbacks=[trends, rec], num_sanity_val_steps=2,
    )
    trainer.fit(_Scripted(), _loader(), _loader())
    return trends, rec.rows


def test_best_is_running_max_and_ignores_sanity_check() -> None:
    _, rows = _fit("max")
    assert [round(r["val/acc_best"], 6) for r in rows] == [0.5, 0.7, 0.7, 0.8]  # not 99


def test_gap_pairs_train_and_val_of_the_same_epoch_and_positive_means_overfitting() -> None:
    _, rows = _fit("max")
    expected = [t - v for t, v in zip(TRAIN, VAL)]  # train(N) - val(N), no one-epoch lag
    assert [round(r["val/overfit_gap"], 6) for r in rows] == [round(e, 6) for e in expected]
    assert all(r["val/overfit_gap"] > 0 for r in rows)


def test_min_mode_orients_the_gap_and_tracks_the_minimum() -> None:
    _, rows = _fit("min")
    assert [round(r["val/acc_best"], 6) for r in rows] == [0.5, 0.5, 0.5, 0.5]  # sanity 99 ignored
    assert round(rows[0]["val/overfit_gap"], 6) == round(VAL[0] - TRAIN[0], 6)  # val - train for min


def test_state_roundtrip_keeps_the_best_for_resumed_runs() -> None:
    trends, _ = _fit("max")
    restored = MetricTrends("val/acc", "max")
    restored.load_state_dict(trends.state_dict())
    assert restored.best == trends.best
    assert abs(restored.best - 0.8) < 1e-6
