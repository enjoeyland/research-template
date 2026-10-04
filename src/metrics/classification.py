"""Classification metrics implementing MetricHandler (minimal set; add more following this pattern)."""

from torchmetrics.classification import MulticlassAccuracy as _TMMulticlassAccuracy
from torchmetrics.classification import MulticlassF1Score as _TMMulticlassF1Score

from src.metrics.metric_base import MetricHandler
from src.utils.model_output import get_field


class _OutputsHandler(MetricHandler):
    """``on_step`` / ``on_epoch_end`` shared by torchmetrics-backed metrics that read two fields of the model
    output.

    Which fields is configuration, not code: ``preds_key`` / ``target_key`` name them (``ModelOutput`` attribute
    or ``extras`` key). A second stream (e.g. ``gt_preds``) therefore is just another metric declared in the
    config with other keys -- no subclass, no ``on_step`` override. One metric object must only ever see ONE
    stream (it accumulates state over the epoch), which is why a second stream is a second object.
    """

    def __init__(self, *args, preds_key: str = "preds", target_key: str = "target", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.preds_key = preds_key
        self.target_key = target_key

    def on_step(self, split, outputs, batch, batch_idx, dataloader_idx=None, *args, **kwargs):
        preds = get_field(outputs, self.preds_key)
        target = get_field(outputs, self.target_key)
        if preds is None or target is None:
            # this stream does not exist at this step (e.g. no second pass at eval): log nothing
            return None
        self.update(preds, target)
        # Return the Metric itself: logged via log_dict(on_epoch=True), Lightning auto-computes and
        # resets it at each real epoch boundary (and isolates it from sanity-check batches) -- no
        # manual .reset() calls needed anywhere (src/metrics/README.md).
        return self

    def on_epoch_end(self, split, *args, **kwargs):
        # None when this metric saw no data this epoch: _add_metric_result skips a None result, so
        # nothing is logged (vs compute() raising). update_called is torchmetrics' own flag, cleared by
        # reset(), so this re-arms each epoch with no manual bookkeeping.
        return self.compute() if self.update_called else None


class MulticlassAccuracy(_OutputsHandler, _TMMulticlassAccuracy):
    """Multiclass accuracy. Inherits the concrete torchmetrics class (not ``Accuracy``) so ``__new__``
    task-dispatch does not break under multiple inheritance with MetricHandler."""


class MulticlassF1Score(_OutputsHandler, _TMMulticlassF1Score):
    """Macro-averaged F1 (torchmetrics default ``average="macro"``)."""
