"""Shell helpers of scripts/sbatch/common (checkpoint_dir, prepare_resume) driven through a real bash."""

import platform
import subprocess
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(platform.system() == "Windows", reason="bash scripts")


def _bash(body: str, env: dict) -> str:
    script = (
        "source scripts/sbatch/common/env.sh 2>/dev/null; source scripts/sbatch/common/resume.sh; "
        + body
    )
    out = subprocess.run(
        ["bash", "-c", script], cwd=ROOT, env=env, capture_output=True, text=True, check=True
    )
    return out.stdout.strip()


def _env(tmp_path: Path, extra_args: str = "") -> dict:
    import os

    env = dict(os.environ)
    env.update(CHECKPOINT_DIR=str(tmp_path), EXTRA_ARGS=extra_args, LOGGER="csv")
    return env


def _real_resume_ckpt(tmp_path: Path, exp: str) -> Path:
    path = tmp_path / "runs" / exp / "checkpoints" / "seed0_resume.ckpt"
    path.parent.mkdir(parents=True)
    with zipfile.ZipFile(path, "w") as z:  # prepare_resume only checks that the file opens as a zip
        z.writestr("x", "y")
    return path


def test_normal_run_resumes_from_the_real_checkpoint(tmp_path) -> None:
    ckpt = _real_resume_ckpt(tmp_path, "exp1")
    out = _bash('prepare_resume exp1 0 exp1-fold0 2>/dev/null; echo "${RESUME_ARGS[*]}"', _env(tmp_path))
    assert f"ckpt_path={ckpt}" in out
    assert (ckpt.parent / "wandb_id_seed0.txt").exists()


def test_smoke_run_never_resumes_and_never_touches_the_real_folder(tmp_path) -> None:
    """debug=smoke runs in its own tree (logs/smoke); a same-named real experiment must not leak in or be polluted."""
    ckpt = _real_resume_ckpt(tmp_path, "exp1")
    before = sorted(p.name for p in ckpt.parent.iterdir())
    out = _bash('prepare_resume exp1 0 exp1-fold0 2>/dev/null; echo "[${RESUME_ARGS[*]}]"', _env(tmp_path, "debug=smoke"))
    assert out == "[]"  # no ckpt_path, no wandb id
    assert sorted(p.name for p in ckpt.parent.iterdir()) == before  # no wandb_id file written into the real run


def test_checkpoint_dir_points_at_the_smoke_tree_for_debug_smoke_only(tmp_path) -> None:
    assert _bash("checkpoint_dir exp1", _env(tmp_path)) == f"{tmp_path}/runs/exp1/checkpoints"
    assert _bash("checkpoint_dir exp1", _env(tmp_path, "debug=smoke")) == "logs/smoke/runs/exp1/checkpoints"
    # other debug configs (callbacks off) keep the normal path; they just skip resume
    assert _bash("checkpoint_dir exp1", _env(tmp_path, "debug=default")) == f"{tmp_path}/runs/exp1/checkpoints"
    out = _bash('prepare_resume exp1 0 n 2>/dev/null; echo "[${RESUME_ARGS[*]}]"', _env(tmp_path, "debug=default"))
    assert out == "[]"
