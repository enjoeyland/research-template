"""Reusable MetricGroup implementations."""

from src.metrics.classification import MulticlassAccuracy, MulticlassF1Score
from src.metrics.metric_base import MetricGroup


class ClassificationMetricGroup(MetricGroup):
    """Train / valid / test multiclass accuracy + macro F1. Add metrics here (or write another group)
    and list the group under ``metric_groups`` in ``configs/metrics/*.yaml``."""

    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        super().__init__()

    def init_metrics(self) -> None:
        for split_dict in (self.metrics_train, self.metrics_valid, self.metrics_test):
            split_dict["acc"] = MulticlassAccuracy(num_classes=self.num_classes)
            split_dict["macro_f1"] = MulticlassF1Score(num_classes=self.num_classes)
