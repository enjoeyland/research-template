"""Classification metrics implementing MetricHandler (minimal set; add more following this pattern)."""

from torchmetrics.classification import MulticlassAccuracy as _TMMulticlassAccuracy

from src.metrics.metric_base import MetricHandler


class MulticlassAccuracy(_TMMulticlassAccuracy, MetricHandler):
    """Multiclass accuracy updated via ``preds`` / ``target`` kwargs.

    Inherits the concrete torchmetrics class (not ``Accuracy``) so ``__new__`` task-dispatch does not
    break under multiple inheritance with MetricHandler.
    """

    def on_step(
        self,
        split,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=None,
        preds=None,
        target=None,
        *args,
        **kwargs,
    ):
        if preds is not None and target is not None:
            self.update(preds, target)
        # Return the Metric itself: logged via log_dict(on_epoch=True), Lightning auto-computes and
        # resets it at each real epoch boundary (and isolates it from sanity-check batches) -- no
        # manual .reset() calls needed anywhere (CLAUDE.md §3).
        return self

    def on_epoch_end(self, split, *args, **kwargs):
        # None when this metric saw no data this epoch: _add_metric_result skips a None result, so
        # nothing is logged (vs compute() raising). update_called is torchmetrics' own flag, cleared by
        # reset(), so this re-arms each epoch with no manual bookkeeping.
        return self.compute() if self.update_called else None
