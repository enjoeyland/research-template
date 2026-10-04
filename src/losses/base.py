"""Loss terms: one ``nn.Module`` per term, composed by ``CompositeLoss`` (src/losses/composite.py).

Contract of a term: ``forward(outputs, batch) -> scalar tensor``.
  * ``outputs`` is the model's ``ModelOutput`` (computed once per step, shared with the metrics),
  * read only what ``requires`` names; take extra state (class priors, margins, ...) from the constructor,
  * do the loss-specific deterministic preprocessing HERE (softmax, logit adjustment, masked mean), not in the model:
    that is what lets a config swap the loss without touching the model,
  * stochastic / expensive values (sampled masks, features) come from ``outputs`` (``extras``) -- never re-sample.

Split a loss out of the model when it has more than one term, is reused across models, or has its own state /
hyperparameters. A one-line cross-entropy can stay inside the module.
"""

from typing import Any, Tuple

import torch
from torch import nn


class LossTerm(nn.Module):
    # fields of the ModelOutput this term reads; CompositeLoss checks them with a clear error
    requires: Tuple[str, ...] = ()

    def forward(self, outputs: Any, batch: Any) -> torch.Tensor:
        raise NotImplementedError
