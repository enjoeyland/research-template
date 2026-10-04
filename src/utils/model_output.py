"""What a model's ``forward`` hands to losses and metrics.

One ``ModelOutput`` per step, computed ONCE and read by both ``loss_fn(outputs, batch)`` and
``metrics.on_step(split, outputs, batch, ...)``. Boundary rule (see CLAUDE.md §3):

  * outputs hold what only the model can produce: network outputs (``logits``), values sampled during
    forward (masks, noise), intermediate features -> ``extras``,
  * deterministic preprocessing (softmax, logit adjustment, masked means, ...) is OWNED BY the loss / metric
    that needs it, so swapping a loss or metric in a config never requires touching the model,
  * a loss / metric names the fields it reads (``requires``, ``preds_key``/``target_key``) and is told clearly
    when the model does not provide them.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

import torch


@dataclass
class ModelOutput:
    logits: Optional[torch.Tensor] = None
    target: Optional[torch.Tensor] = None
    # defaults to argmax(logits); set explicitly for non-classification outputs
    preds: Optional[torch.Tensor] = None
    # model-specific values other terms may read (sampled masks, features, second-pass predictions, ...)
    extras: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.preds is None and self.logits is not None:
            self.preds = self.logits.argmax(dim=-1)

    def get(self, key: str, default: Any = None) -> Any:
        """Field lookup by name: dataclass fields first, then ``extras``."""
        value = getattr(self, key, None)
        return value if value is not None else self.extras.get(key, default)


def get_field(outputs: Any, key: str, default: Any = None) -> Any:
    """``outputs[key]`` for a ``ModelOutput`` or a plain ``Mapping`` (so plain dicts keep working)."""
    if outputs is None:
        return default
    if isinstance(outputs, Mapping):
        value = outputs.get(key)
        return default if value is None else value
    return outputs.get(key, default)
