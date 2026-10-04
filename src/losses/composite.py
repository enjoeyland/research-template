"""Weighted sum of named loss terms, configured from Hydra."""

from typing import Any, Dict, Mapping

import torch
from torch import nn

from src.losses.base import LossTerm
from src.utils.model_output import get_field


class CompositeLoss(nn.Module):
    """``terms = {name: {"weight": w, "term": LossTerm}}`` (see configs/losses/*.yaml).

    Returns ``{"loss": sum_i w_i * term_i, "loss_<name>": term_i (unweighted), ...}`` -- every entry is
    ready for ``self.log_dict({f"{split}/{k}": v ...})``. The weighted total is what to backprop; the
    unweighted terms are what to plot. A term with weight 0 is skipped (not computed, not logged).
    Diagnostics (e.g. an entropy floor) are NOT loss terms: log them as metrics.
    """

    def __init__(self, terms: Mapping[str, Mapping[str, Any]]) -> None:
        super().__init__()
        assert len(terms) > 0, "CompositeLoss needs at least one term"
        self.weights: Dict[str, float] = {}
        self.terms = nn.ModuleDict()
        for name, spec in terms.items():
            assert isinstance(spec["term"], LossTerm), f"terms.{name}.term must be a LossTerm"
            self.terms[name] = spec["term"]
            self.weights[name] = float(spec.get("weight", 1.0))
        self._checked = False

    def _check_requires(self, outputs: Any) -> None:
        for name, term in self.terms.items():
            if self.weights[name] == 0.0:
                continue
            missing = [k for k in term.requires if get_field(outputs, k) is None]
            if missing:
                raise KeyError(
                    f"loss term '{name}' ({type(term).__name__}) requires {missing} in the model output, "
                    f"but the model did not provide them (put them in ModelOutput / ModelOutput.extras)"
                )
        self._checked = True

    def forward(self, outputs: Any, batch: Any) -> Dict[str, torch.Tensor]:
        if not self._checked:
            self._check_requires(outputs)
        total = None
        result: Dict[str, torch.Tensor] = {}
        for name, term in self.terms.items():
            w = self.weights[name]
            if w == 0.0:
                continue
            value = term(outputs, batch)
            result[f"loss_{name}"] = value
            total = w * value if total is None else total + w * value
        assert total is not None, "all loss terms have weight 0"
        return {"loss": total, **result}
