"""Terms that wrap or read model-provided fields (generic building blocks, found necessary when porting MCCG)."""

from typing import Any, Mapping

import torch

from src.losses.base import LossTerm
from src.utils.model_output import get_field


class FieldTerm(LossTerm):
    """A loss term the MODEL already computed (e.g. a KL / orthogonality regularizer that needs decoder
    internals): reads the scalar ``outputs[key]`` so it takes part in the weighted sum like any other term."""

    def __init__(self, key: str) -> None:
        super().__init__()
        self.key = key
        self.requires = (key,)

    def forward(self, outputs: Any, batch: Any) -> torch.Tensor:
        return get_field(outputs, self.key)


class PassBlend(LossTerm):
    """Blend ``term`` over several forward passes of the same step (e.g. a ground-truth pass and a self-predicted
    pass): ``sum_p weights[p] * term(passes[p], batch)``.

    The model puts one ``ModelOutput`` per pass into ``outputs.extras["passes"] = {"gt": ..., "self": ...}``.
    Without ``passes`` (e.g. evaluation, where there is a single pass) the term is applied to ``outputs`` itself.
    Nested blends are linear, so they flatten into one weight set:
    ``w*L(gt) + (1-w)*((1-w2)*L(self) + w2*L(noexo))`` == weights ``{gt: w, self: (1-w)(1-w2), noexo: (1-w)w2}``.
    Weights are used as given (not normalized).
    """

    def __init__(self, term: LossTerm, weights: Mapping[str, float]) -> None:
        super().__init__()
        self.term = term
        self.weights = {str(k): float(v) for k, v in weights.items()}

    def forward(self, outputs: Any, batch: Any) -> torch.Tensor:
        passes = get_field(outputs, "passes")
        if passes is None:
            return self.term(outputs, batch)
        total = None
        for name, w in self.weights.items():
            if w == 0.0:
                continue
            if name not in passes:
                raise KeyError(f"PassBlend: weight given for pass '{name}' but the model output has passes {list(passes)}")
            value = w * self.term(passes[name], batch)
            total = value if total is None else total + value
        return total
