"""Base classes for composable task metrics (ported from AttnHeads_PhaseTrans via Medical-CausalInference).

Usage in a LightningModule (see src/models/toy_module.py for plain torchmetrics, tests/test_metrics.py for this API):
  * ``on_step(split, ...)`` returns the metric OBJECTS -> pass them to ``self.log_dict(..., on_epoch=True)``;
    Lightning then computes and resets them at each epoch boundary (never call ``.reset()`` by hand, CLAUDE.md §3),
  * call ``self.metrics.reset_best()`` once from ``on_train_start``,
  * ``self.metrics.on_epoch_end(split)`` returns the epoch values (+ ``val/<name>_best``, ``val/overfit_gap``).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional

from torch.nn import Module, ModuleDict, ModuleList
from torchmetrics import MaxMetric, MinMetric

Split = Literal["train", "valid", "test"]


class MetricHandler:
    """Protocol-like base for metrics with step / epoch-end hooks."""

    def on_step(
        self,
        split: Split,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=None,
        *args,
        **kwargs,
    ):
        """Update metric state for a batch. Return a scalar/dict to log, or None."""
        return self

    def on_epoch_end(self, split: Split, *args, **kwargs):
        """Compute and return the final metric value (scalar or dict), or None."""
        return None


def _add_metric_result(
    metrics: Dict[str, Any],
    split: str,
    name: str,
    result: Any,
    dataloader_idx: Optional[int] = None,
) -> None:
    """Flatten a metric result into ``{split}/{name}`` (``valid`` → ``val``)."""
    if result is None:
        return
    split_key = "val" if split == "valid" else split
    prefix = f"{split_key}/d{dataloader_idx}" if dataloader_idx is not None else split_key

    if isinstance(result, dict):
        for key, value in result.items():
            metrics[f"{prefix}/{name}/{key}"] = value
    else:
        metrics[f"{prefix}/{name}"] = result


class MetricGroup(Module, MetricHandler, ABC):
    """Group of related metrics, split into train / valid / test ModuleDicts."""

    def __init__(self) -> None:
        super().__init__()
        self.metrics_train = ModuleDict()
        self.metrics_valid = ModuleDict()
        self.metrics_test = ModuleDict()
        self.init_metrics()

    @abstractmethod
    def init_metrics(self) -> None:
        """Populate ``metrics_train`` / ``metrics_valid`` / ``metrics_test``."""

    @property
    def _metrics_dict(self) -> Dict[Split, ModuleDict]:
        return {
            "train": self.metrics_train,
            "valid": self.metrics_valid,
            "test": self.metrics_test,
        }

    @property
    def metrics(self) -> ModuleList:
        return ModuleList([self.metrics_train, self.metrics_valid, self.metrics_test])

    def reset(self, split: Optional[Split] = None) -> None:
        splits: List[Split] = [split] if split is not None else ["train", "valid", "test"]
        for s in splits:
            for metric in self._metrics_dict[s].values():
                if hasattr(metric, "reset"):
                    metric.reset()

    def on_step(
        self, split, outputs, batch, batch_idx, dataloader_idx=None, *args, **kwargs
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for name, metric in self._metrics_dict[split].items():
            try:
                result = metric.on_step(
                    split, outputs, batch, batch_idx, dataloader_idx, *args, **kwargs
                )
                _add_metric_result(out, split, name, result, dataloader_idx)
            except Exception as e:
                raise RuntimeError(f"Error in metric {name}: {e}") from e
        return out

    def on_epoch_end(self, split, *args, **kwargs) -> Dict[str, Any]:
        # No manual .reset() here: on_step returns `self` for each metric, so as long as it's
        # logged via log_dict(on_epoch=True) somewhere, Lightning's own log()-based mechanism
        # already computes and resets each metric at the real epoch boundary (and isolates it
        # from sanity-check batches). Resetting here too would race with that and produce a
        # (harmless but noisy) "compute called before update" warning.
        out: Dict[str, Any] = {}
        for name, metric in self._metrics_dict[split].items():
            try:
                result = metric.on_epoch_end(split, *args, **kwargs)
                _add_metric_result(out, split, name, result, None)
            except Exception as e:
                raise RuntimeError(f"Error in metric {name}: {e}") from e
        return out


class TaskMetrics(Module, MetricHandler):
    """Compose MetricGroups; expose ``monitor_metric`` / ``monitor_mode`` for checkpoint / early-stopping.

    Also owns two cross-epoch / cross-split trackers derived from ``monitor_metric``, computed
    automatically inside ``on_epoch_end``:

    * ``val/<name>_best`` -- best-so-far value of the monitored quantity on ``valid`` (e.g.
      ``val/macro_f1_best`` when ``monitor_metric="val/macro_f1"``),
    * ``val/overfit_gap`` -- train-vs-val gap of that same quantity, ALWAYS oriented so that positive =
      overfitting whichever direction the quantity improves in (train - val for ``max``, val - train
      for ``min``). Near zero / negative means generalizing normally.

    Models only call ``self.metrics.on_epoch_end(split)`` plus ``reset_best()`` once from
    ``on_train_start`` (see its docstring for why that is separate from ``reset()``).

    CAVEAT the gap cannot detect: it is only a generalization measure if train and val are measured
    under the SAME input conditions. If training masks/augments inputs differently from validation, a
    large positive gap is partly just the protocol difference."""

    def __init__(
        self,
        metric_groups: List[MetricGroup],
        monitor_metric: str = "val/acc",
        monitor_mode: str = "max",
    ) -> None:
        super().__init__()
        self.metric_groups = ModuleList(metric_groups)
        self.monitor_metric = monitor_metric
        self.monitor_mode = monitor_mode
        # e.g. "macro_f1" from "val/macro_f1" -- used to look up this SAME quantity under train/
        # test split prefixes too (monitor_metric itself is always a "val/..." key).
        self._monitor_name = monitor_metric.split("/", 1)[-1]
        # ONE reading of monitor_mode, shared by both derived trackers below (reading it twice
        # independently once left overfit_gap with an inverted sign) -- keep them tied to this flag.
        self._higher_is_better = monitor_mode == "max"
        self.best_score = MaxMetric() if self._higher_is_better else MinMetric()
        self._last_train_score: Optional[float] = None

    @property
    def metrics(self) -> ModuleList:
        return self.metric_groups

    def reset(self, split: Optional[Split] = None) -> None:
        for group in self.metric_groups:
            group.reset(split)

    def reset_best(self) -> None:
        """Call once from ``on_train_start`` -- resets the cross-RUN best-score tracker and the
        train-score cache. Deliberately NOT part of ``reset()`` and NOT auto-reset the way the
        per-step ``MetricGroup`` metrics are (CLAUDE.md §3: that warning is about THOSE metrics, which
        are logged as the metric object so Lightning resets them at each epoch boundary).
        ``best_score`` / ``_last_train_score`` are plain trackers this class owns and logs as
        already-computed scalars, so nothing auto-resets them -- they need exactly one manual reset
        per training run, here.

        Also wipes whatever the pre-training sanity-check validation pass wrote into ``best_score``:
        Lightning runs that sanity check (a real ``on_validation_epoch_end`` on random-init weights)
        BEFORE ``on_train_start``, so without this reset its score could linger as a spuriously
        good/bad "best" for the whole run."""
        self.best_score.reset()
        self._last_train_score = None

    def on_step(
        self, split, outputs, batch, batch_idx, dataloader_idx=None, *args, **kwargs
    ) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {}
        for group in self.metric_groups:
            metrics.update(
                group.on_step(split, outputs, batch, batch_idx, dataloader_idx, *args, **kwargs)
            )
        return metrics

    def on_epoch_end(self, split, *args, **kwargs) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {}
        for group in self.metric_groups:
            metrics.update(group.on_epoch_end(split, *args, **kwargs))

        monitor_score = self.compute_monitor_score(metrics)
        _add_metric_result(metrics, split, "monitor_score", monitor_score, None)

        if split == "train":
            # cached for the "valid" branch below -- on_train_epoch_end fires before
            # on_validation_epoch_end within the same epoch (Lightning's default ordering), so a
            # real (post-sanity-check) "valid" call always sees that same epoch's value.
            self._last_train_score = metrics.get(f"train/{self._monitor_name}")
        elif split == "valid" and monitor_score is not None:
            self.best_score(monitor_score)
            metrics[f"val/{self._monitor_name}_best"] = self.best_score.compute()
            # None on sanity-check validation (runs before any training epoch) -- not logged then,
            # rather than a misleading 0/NaN.
            if self._last_train_score is not None:
                # Oriented by monitor_mode so POSITIVE ALWAYS MEANS OVERFITTING: for a "max"
                # quantity (f1/acc) overfitting shows up as train ABOVE val, for a "min" one (a
                # loss) as val ABOVE train.
                metrics["val/overfit_gap"] = (
                    self._last_train_score - monitor_score
                    if self._higher_is_better
                    else monitor_score - self._last_train_score
                )

        return metrics

    def compute_monitor_score(self, metrics: Dict[str, Any]) -> Optional[float]:
        return metrics.get(self.monitor_metric)
