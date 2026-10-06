"""Conditional skipping of tests (trimmed from PyTorchLightning's tests/helpers/runif.py to what this repo uses)."""

from typing import Any, Dict

import pytest
import torch
from pytest import MarkDecorator

from tests.helpers.package_available import _SH_AVAILABLE, _WANDB_AVAILABLE


class RunIf:
    """RunIf wrapper for conditional skipping of tests. Fully compatible with `@pytest.mark`.

    Example:

    ```python
        @RunIf(min_gpus=1)
        def test_on_gpu():
            ...
    ```
    """

    def __new__(
        cls,
        min_gpus: int = 0,
        sh: bool = False,
        wandb: bool = False,
        **kwargs: Dict[Any, Any],
    ) -> MarkDecorator:
        """Creates a new `@RunIf` `MarkDecorator` decorator.

        :param min_gpus: Min number of GPUs required to run test.
        :param sh: If the `sh` module is required to run the test.
        :param wandb: If the `wandb` module is required to run the test.
        :param kwargs: Native `pytest.mark.skipif` keyword arguments.
        """
        conditions = []
        reasons = []

        if min_gpus:
            conditions.append(torch.cuda.device_count() < min_gpus)
            reasons.append(f"GPUs>={min_gpus}")

        if sh:
            conditions.append(not _SH_AVAILABLE)
            reasons.append("sh")

        if wandb:
            conditions.append(not _WANDB_AVAILABLE)
            reasons.append("wandb")

        reasons = [rs for cond, rs in zip(conditions, reasons) if cond]
        return pytest.mark.skipif(
            condition=any(conditions),
            reason=f"Requires: [{' + '.join(reasons)}]",
            **kwargs,
        )
