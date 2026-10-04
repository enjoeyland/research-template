"""Classification loss terms."""

import torch
import torch.nn.functional as F

from src.losses.base import LossTerm
from src.utils.model_output import get_field


class CrossEntropy(LossTerm):
    requires = ("logits", "target")

    def __init__(self, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(self, outputs, batch) -> torch.Tensor:
        return F.cross_entropy(
            get_field(outputs, "logits"),
            get_field(outputs, "target"),
            label_smoothing=self.label_smoothing,
        )
