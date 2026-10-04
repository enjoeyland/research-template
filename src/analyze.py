"""Post-hoc analysis entry point (skeleton).

Loads the checkpoints of one experiment (``seed*_last.ckpt`` under ``paths.ckpt_dir``) and writes
tables / plots to ``logs/analyze/runs/<experiment_name>/``. Put reusable analysis code in
``src/analysis/<YYMMDD_topic>/`` (see src/analysis/README.md) and call it from here.

    python src/analyze.py experiment/train=<name> seeds=[0,1,2]
"""

from pathlib import Path
from typing import List

import hydra
import rootutils
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import RankedLogger, extras  # noqa: E402

log = RankedLogger(__name__, rank_zero_only=True)


@hydra.main(version_base="1.3", config_path="../configs", config_name="analyze.yaml")
def main(cfg: DictConfig) -> None:
    extras(cfg)
    out_dir = Path(cfg.paths.output_dir)
    seeds: List[int] = list(cfg.seeds)
    for seed in seeds:
        ckpt = Path(cfg.paths.ckpt_dir) / f"seed{seed}_last.ckpt"
        log.info(f"TODO: analyze {ckpt} -> {out_dir}")


if __name__ == "__main__":
    main()
