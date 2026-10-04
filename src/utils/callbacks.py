"""Custom Lightning callbacks.

``NamedLastModelCheckpoint``: plain ``ModelCheckpoint`` writes its "last" checkpoint to the FIXED
filename ``CHECKPOINT_NAME_LAST = "last"`` (a class attribute, not an ``__init__`` kwarg). When several
runs (seeds, folds) share one ``dirpath`` they would all overwrite the same ``last.ckpt``. This subclass
makes it a constructor kwarg so it can be set per run from Hydra, e.g.
``checkpoint_name_last: "seed${seed}_last"``.
"""

from typing import Optional

from lightning.pytorch.callbacks import ModelCheckpoint


class NamedLastModelCheckpoint(ModelCheckpoint):
    def __init__(self, checkpoint_name_last: Optional[str] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        if checkpoint_name_last is not None:
            self.CHECKPOINT_NAME_LAST = checkpoint_name_last
