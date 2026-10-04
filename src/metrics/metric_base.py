"""Base classes for composable task metrics (ported from AttnHeads_PhaseTrans via Medical-CausalInference).

Usage in a LightningModule (see src/models/toy_module.py for plain torchmetrics, tests/test_metrics.py for this API):
  * ``on_step(split, ...)`` returns the metric OBJECTS -> pass them to ``self.log_dict(..., on_epoch=True)``;
    Lightning then computes and resets them at each epoch boundary (never call ``.reset()`` by hand, src/metrics/README.md),
  * ``self.metrics.on_epoch_end(split)`` returns the epoch values.
  * ``val/<name>_best`` and ``val/overfit_gap`` curves come from the ``MetricTrends`` callback (src/utils/callbacks.py).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional

from torch.nn import Module, ModuleDict, ModuleList

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
    """Compose MetricGroups; expose ``monitor_metric`` / ``monitor_mode`` for checkpointing, early stopping and
    the ``MetricTrends`` callback (best-so-far and overfit-gap curves live there, not here)."""

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

    @property
    def metrics(self) -> ModuleList:
        return self.metric_groups

    def reset(self, split: Optional[Split] = None) -> None:
        for group in self.metric_groups:
            group.reset(split)

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
        return metrics
