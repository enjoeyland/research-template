"""Custom Lightning callbacks.

``NamedLastModelCheckpoint``: plain ``ModelCheckpoint`` writes its "last" checkpoint to the FIXED
filename ``CHECKPOINT_NAME_LAST = "last"`` (a class attribute, not an ``__init__`` kwarg). When several
runs (seeds, folds) share one ``dirpath`` they would all overwrite the same ``last.ckpt``. This subclass
makes it a constructor kwarg so it can be set per run from Hydra, e.g.
``checkpoint_name_last: "seed${seed}_last"``.
"""

from typing import Optional

from lightning.pytorch.callbacks import Callback, ModelCheckpoint


class NamedLastModelCheckpoint(ModelCheckpoint):
    def __init__(self, checkpoint_name_last: Optional[str] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        if checkpoint_name_last is not None:
            self.CHECKPOINT_NAME_LAST = checkpoint_name_last


class MetricTrends(Callback):
    """Per-epoch curves derived from the monitored metric, for plotting in wandb / the csv logger:

      * ``<monitor>_best``  -- best value so far (e.g. ``val/acc_best``), a running max/min,
      * ``val/overfit_gap`` -- train value minus val value of the SAME epoch (``max`` monitor; val minus train
        for ``min``), i.e. POSITIVE = overfitting whichever direction the metric improves in.

    Why a callback and not part of ``TaskMetrics``: both curves are derived from values that are already
    logged, so nothing needs to live in the metric / model code (no ``reset_best()`` call, no extra state in
    the module). It runs in ``on_train_epoch_end``, which Lightning fires AFTER the validation loop of the same
    epoch, so ``callback_metrics`` holds train AND val of that epoch -- reading them from a validation hook
    pairs val(N) with the stale train(N-1). The running best is part of the checkpoint (``state_dict``), so a
    resumed run continues the curve instead of restarting it. The sanity-check validation never reaches
    ``on_train_epoch_end``, so it cannot pollute the best.

    With ``check_val_every_n_epoch > 1`` the val value is stale between validations; the curves are then only
    updated on epochs where the monitored value changed.

    For wandb the monitored metric also gets ``define_metric(summary=<mode>)``, so the run table / summary shows
    the best value without any code on the reader's side.
    """

    def __init__(self, monitor: str, mode: str = "max") -> None:
        assert mode in ("max", "min"), mode
        assert monitor.startswith("val/"), f"monitor must be a 'val/...' metric, got {monitor!r}"
        self.monitor, self.mode = monitor, mode
        self.train_key = "train/" + monitor[len("val/") :]
        self.best: Optional[float] = None
        self._last_val: Optional[float] = None

    def on_fit_start(self, trainer, pl_module) -> None:
        for logger in trainer.loggers:
            experiment = getattr(logger, "experiment", None)
            if hasattr(experiment, "define_metric"):  # wandb run
                experiment.define_metric(self.monitor, summary=self.mode)

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        metrics = trainer.callback_metrics
        if self.monitor not in metrics:
            return
        val = float(metrics[self.monitor])
        if val == self._last_val and trainer.current_epoch % trainer.check_val_every_n_epoch != 0:
            return  # val was not re-evaluated this epoch
        self._last_val = val
        better = (lambda a, b: a > b) if self.mode == "max" else (lambda a, b: a < b)
        self.best = val if self.best is None or better(val, self.best) else self.best
        pl_module.log(f"{self.monitor}_best", self.best, sync_dist=True)
        if self.train_key in metrics:
            train = float(metrics[self.train_key])
            gap = train - val if self.mode == "max" else val - train
            pl_module.log("val/overfit_gap", gap, sync_dist=True)

    def state_dict(self) -> dict:
        return {"best": self.best, "last_val": self._last_val}

    def load_state_dict(self, state_dict: dict) -> None:
        self.best = state_dict.get("best")
        self._last_val = state_dict.get("last_val")


class ResumeModelCheckpoint(ModelCheckpoint):
    """The mid-run resume checkpoint (``configs/callbacks/model_checkpoint_resume.yaml``).

    Lightning derives a ``ModelCheckpoint``'s ``state_key`` from (monitor, mode, every_n_train_steps, every_n_epochs,
    train_time_interval) and refuses to start when two of them are equal. The "last" checkpoint saves every
    ``trainer.max_epochs`` epochs and this one every 10, so for ``max_epochs == 10`` the keys collided
    (``RuntimeError: Found more than one stateful callback of type ModelCheckpoint``) -- found with the toy example, the
    only config whose max_epochs equals the resume interval. A fixed, distinct key removes the collision for any
    max_epochs.
    """

    @property
    def state_key(self) -> str:
        return "ModelCheckpoint{resume}"
