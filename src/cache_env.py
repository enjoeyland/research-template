"""Default cache locations on the disposable disk (call this BEFORE importing torch / transformers / timm).

Cluster storage convention (CLAUDE.md §2):
  * /lustre/<user>/   -- kept long-term: checkpoints (``CHECKPOINT_DIR``), results worth keeping,
  * /scratch2/<user>/ -- may be wiped: caches, tmp, venvs.

Pretrained weights and downloaded datasets are re-downloadable caches, so they go to scratch2 instead of
``~/.cache`` (home is small). ``set_cache_defaults()`` only fills variables that are NOT already set, so the
environment / ``.env`` always wins (rootutils loads ``.env`` before this runs). Root: ``CACHE_DIR`` if set, else
``/scratch2/$USER/cache`` when that directory's parent exists, else nothing changes.

Libraries read these variables at import time (huggingface_hub) or call time (torch.hub), which is why the entry
points call this first, right after ``rootutils.setup_root``.
"""

import getpass
import os
from typing import Dict, Optional

# variable -> sub directory of the cache root
CACHE_VARS: Dict[str, str] = {
    "HF_HOME": "huggingface",  # transformers / datasets / huggingface_hub (timm weights come through here too)
    "TORCH_HOME": "torch",  # torch.hub, torchvision pretrained weights
    "TORCH_EXTENSIONS_DIR": "torch_extensions",
    "TRITON_CACHE_DIR": "triton",
    "XDG_CACHE_HOME": "xdg",  # generic fallback for libraries that follow the XDG spec
    "WANDB_CACHE_DIR": "wandb",
    "WANDB_DATA_DIR": "wandb/data",
    "MPLCONFIGDIR": "matplotlib",
    "PIP_CACHE_DIR": "pip",
}


def default_cache_root() -> Optional[str]:
    root = os.environ.get("CACHE_DIR")
    if root:
        return root
    scratch = os.path.join("/scratch2", os.environ.get("USER") or getpass.getuser())
    return os.path.join(scratch, "cache") if os.path.isdir(scratch) else None


def set_cache_defaults() -> Optional[str]:
    """Fill the unset cache variables under the cache root; returns the root used (None = nothing changed)."""
    root = default_cache_root()
    if root is None:
        return None
    for var, sub in CACHE_VARS.items():
        os.environ.setdefault(var, os.path.join(root, sub))
    return root
