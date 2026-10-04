import pytest
import torch

from src.losses import CompositeLoss, CrossEntropy, LossTerm
from src.utils.model_output import ModelOutput


class _Const(LossTerm):
    """A term returning a fixed value, to test the composition arithmetic."""

    def __init__(self, value: float, requires=()) -> None:
        super().__init__()
        self.value = value
        self.requires = tuple(requires)

    def forward(self, outputs, batch):
        return torch.tensor(self.value)


def _outputs() -> ModelOutput:
    return ModelOutput(logits=torch.tensor([[2.0, 0.0], [0.0, 2.0]]), target=torch.tensor([0, 1]))


def test_composite_weighted_sum_and_keys() -> None:
    loss = CompositeLoss({"a": {"weight": 2.0, "term": _Const(1.0)}, "b": {"weight": 0.5, "term": _Const(4.0)}})
    out = loss(_outputs(), None)
    assert float(out["loss"]) == pytest.approx(2.0 * 1.0 + 0.5 * 4.0)
    # terms are logged UNweighted, the total is weighted
    assert float(out["loss_a"]) == 1.0 and float(out["loss_b"]) == 4.0


def test_zero_weight_term_is_skipped() -> None:
    loss = CompositeLoss({"a": {"weight": 1.0, "term": _Const(1.0)}, "off": {"weight": 0.0, "term": _Const(9.0)}})
    out = loss(_outputs(), None)
    assert "loss_off" not in out and float(out["loss"]) == 1.0


def test_missing_required_field_gives_a_clear_error() -> None:
    loss = CompositeLoss({"z": {"weight": 1.0, "term": _Const(1.0, requires=("z",))}})
    with pytest.raises(KeyError, match="requires \\['z'\\]"):
        loss(_outputs(), None)
    # provided through extras -> fine
    out = _outputs()
    out.extras["z"] = torch.zeros(1)
    assert float(loss(out, None)["loss"]) == 1.0


def test_cross_entropy_matches_torch() -> None:
    outputs = _outputs()
    expected = torch.nn.functional.cross_entropy(outputs.logits, outputs.target)
    assert float(CrossEntropy()(outputs, None)) == pytest.approx(float(expected))


def test_pass_blend_weights_passes_and_falls_back_without_passes() -> None:
    from src.losses import PassBlend

    gt = ModelOutput(logits=torch.tensor([[5.0, 0.0]]), target=torch.tensor([0]))
    slf = ModelOutput(logits=torch.tensor([[0.0, 5.0]]), target=torch.tensor([0]))
    ce = CrossEntropy()
    blend = PassBlend(ce, {"gt": 0.25, "self": 0.75})
    out = ModelOutput(logits=slf.logits, target=slf.target, extras={"passes": {"gt": gt, "self": slf}})
    expected = 0.25 * float(ce(gt, None)) + 0.75 * float(ce(slf, None))
    assert float(blend(out, None)) == pytest.approx(expected)
    # eval: a single pass, no `passes` -> the term itself
    assert float(blend(slf, None)) == pytest.approx(float(ce(slf, None)))
    with pytest.raises(KeyError, match="noexo"):
        PassBlend(ce, {"noexo": 1.0})(out, None)


def test_field_term_reads_a_model_computed_scalar() -> None:
    from src.losses import FieldTerm

    out = ModelOutput(extras={"kl": torch.tensor(0.7)})
    loss = CompositeLoss({"kl": {"weight": 0.1, "term": FieldTerm("kl")}})
    assert float(loss(out, None)["loss"]) == pytest.approx(0.07)
    with pytest.raises(KeyError, match="requires \\['kl'\\]"):
        loss(ModelOutput(), None)
