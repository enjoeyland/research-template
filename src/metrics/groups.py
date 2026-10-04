"""Reusable MetricGroup implementations."""

from typing import Sequence

from src.metrics.classification import MulticlassAccuracy, MulticlassF1Score
from src.metrics.metric_base import MetricGroup
from src.metrics.scalar import FieldMean


class ClassificationMetricGroup(MetricGroup):
    """Train / valid / test multiclass accuracy + macro F1, read from ``outputs[preds_key]`` / ``outputs[target_key]``.

    To measure a second prediction stream alongside the first, declare another group in the config with other
    keys and a ``name_prefix`` (so the logged names do not collide):

        - _target_: src.metrics.ClassificationMetricGroup
          num_classes: 3
        - _target_: src.metrics.ClassificationMetricGroup
          num_classes: 3
          name_prefix: "gt/"
          preds_key: gt_preds

    A group whose fields are absent at some step (e.g. no second pass at eval) simply logs nothing there.
    Add metrics here (or write another group) and list the group under ``metric_groups`` in configs/metrics/*.yaml.
    """

    def __init__(
        self,
        num_classes: int,
        preds_key: str = "preds",
        target_key: str = "target",
        name_prefix: str = "",
    ) -> None:
        self.num_classes = num_classes
        self.preds_key = preds_key
        self.target_key = target_key
        self.name_prefix = name_prefix
        super().__init__()

    def init_metrics(self) -> None:
        keys = dict(preds_key=self.preds_key, target_key=self.target_key)
        for split_dict in (self.metrics_train, self.metrics_valid, self.metrics_test):
            split_dict[f"{self.name_prefix}acc"] = MulticlassAccuracy(num_classes=self.num_classes, **keys)
            split_dict[f"{self.name_prefix}macro_f1"] = MulticlassF1Score(num_classes=self.num_classes, **keys)


class FieldMeanGroup(MetricGroup):
    """Epoch means of scalar diagnostics the model puts into its output (``FieldMean`` per key), logged as
    ``<split>/<name_prefix><key>``. Diagnostics belong here, not in the loss."""

    def __init__(self, keys: Sequence[str], name_prefix: str = "diag/") -> None:
        self.keys = list(keys)
        self.name_prefix = name_prefix
        super().__init__()

    def init_metrics(self) -> None:
        for split_dict in (self.metrics_train, self.metrics_valid, self.metrics_test):
            for key in self.keys:
                split_dict[f"{self.name_prefix}{key}"] = FieldMean(key)
