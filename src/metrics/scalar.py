"""Metrics over scalar diagnostics the model puts into its output."""

from torchmetrics import MeanMetric

from src.metrics.metric_base import MetricHandler
from src.utils.model_output import get_field


class FieldMean(MeanMetric, MetricHandler):
    """Epoch mean of the scalar ``outputs[key]`` (e.g. an entropy floor, a gate temperature).

    Diagnostics are measurements, not loss terms: log them through a metric so they never enter the loss sum or a
    pass blend. Absent at a step (e.g. a training-only value at eval) -> nothing is logged."""

    def __init__(self, key: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.key = key

    def on_step(self, split, outputs, batch, batch_idx, dataloader_idx=None, *args, **kwargs):
        value = get_field(outputs, self.key)
        if value is None:
            return None
        self.update(value.detach())
        return self

    def on_epoch_end(self, split, *args, **kwargs):
        return self.compute() if self.update_called else None
