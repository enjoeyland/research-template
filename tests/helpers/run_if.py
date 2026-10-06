"""Conditional skipping of tests (trimmed from PyTorchLightning's tests/helpers/runif.py to what this repo uses)."""

from typing import Any, Dict

import pytest
import torch
from pytest import MarkDecorator


class RunIf:
    """RunIf wrapper for conditional skipping of tests. Fully compatible with `@pytest.mark`.

    Example:

    ```python
        @RunIf(min_gpus=1)
        def test_on_gpu():
            ...
    ```
    """

    def __new__(cls, min_gpus: int = 0, **kwargs: Dict[Any, Any]) -> MarkDecorator:
        """Creates a new `@RunIf` `MarkDecorator` decorator.

        :param min_gpus: Min number of GPUs required to run test.
        :param kwargs: Native `pytest.mark.skipif` keyword arguments.
        """
        have = torch.cuda.device_count()
        return pytest.mark.skipif(
            condition=have < min_gpus,
            reason=f"Requires: [GPUs>={min_gpus}] (found {have})",
            **kwargs,
        )
