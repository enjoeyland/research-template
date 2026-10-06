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


def _preflight(body: str, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    import os

    env = dict(os.environ)
    env.update(env_extra or {})
    return subprocess.run(
        ["bash", "-c", "source scripts/sbatch/common/preflight.sh; " + body],
        cwd=ROOT, env=env, capture_output=True, text=True,
    )


@pytest.mark.parametrize(
    "spec, expected",
    [("", 1), ("0-3", 4), ("0-14%4", 4), ("0-14%20", 15), ("0,2,5", 3), ("0,2,5%2", 2), ("0-9:2", 5)],
)
def test_array_concurrency(spec, expected) -> None:
    assert _preflight(f'array_concurrency "{spec}"').stdout.strip() == str(expected)


@pytest.mark.parametrize(
    "value, seconds",
    [("03:00:00", 10800), ("24:00:00", 86400), ("1-00:00:00", 86400), ("30:00", 1800), ("45", 2700)],
)
def test_time_to_seconds(value, seconds) -> None:
    assert _preflight(f'time_to_seconds "{value}"').stdout.strip() == str(seconds)


_GPU48 = {"PROFILE_NAME": "gpu48", "SLURM_WARN_CONCURRENCY": "4", "SLURM_WARN_AFTER": "03:00:00"}


@pytest.mark.parametrize(
    "array, time_limit, warns",
    [
        ("0-9%4", "24:00:00", True),   # 4 at once and longer than 3h
        ("0-9%3", "24:00:00", False),  # only 3 at once
        ("0-9%4", "03:00:00", False),  # exactly 3h is not longer than 3h
        ("0-3", "05:00:00", True),     # no throttle: all 4 tasks run at once
        ("", "24:00:00", False),       # a single job
    ],
)
def test_concurrency_warning_only_when_both_conditions_hold(array, time_limit, warns) -> None:
    out = _preflight(f'preflight_warn_concurrency "{array}" "{time_limit}"', _GPU48)
    assert ("WARNING" in out.stderr) == warns, out.stderr


def test_no_warning_without_profile_knobs() -> None:
    out = _preflight('preflight_warn_concurrency "0-9%8" "24:00:00"', {"PROFILE_NAME": "gpu24"})
    assert out.stderr == ""


def test_profile_knobs_do_not_leak_between_profiles() -> None:
    out = _preflight("SLURM_CHECK_START=1; SLURM_WARN_AFTER=03:00:00; preflight_reset_knobs; echo \"[${SLURM_CHECK_START:-}${SLURM_WARN_AFTER:-}]\"")
    assert out.stdout.strip() == "[]"
