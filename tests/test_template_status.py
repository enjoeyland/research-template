"""`make template-status` (src/template_status.py) and `make rename`'s handling of the template-only folder."""

import platform
import shutil
import subprocess
from pathlib import Path

import pytest

from src import template_status as ts

ROOT = Path(__file__).resolve().parents[1]

CHANGELOG = """# Template 변경 이력

## v3 (2026-10-06) — 세 번째
- 추가: C
- 다운스트림에서 할 일: c를 가져온다.
- 영향 경로: `c/`

## v2 (2026-10-05) — 두 번째
- 추가: B

## v1 (2026-10-04)
- 추가: A
"""


def test_parse_changelog_orders_newest_first_and_keeps_bodies() -> None:
    entries = ts.parse_changelog(CHANGELOG)
    assert [e.version for e in entries] == [3, 2, 1]
    assert entries[0].title == "세 번째" and entries[2].title == ""
    assert "다운스트림에서 할 일" in entries[0].body and "추가: A" in entries[2].body


@pytest.mark.parametrize(
    "url",
    ["https://github.com/o/r", "https://github.com/o/r.git", "git@github.com:o/r.git"],
)
def test_changelog_url_accepts_https_and_ssh_github_urls(url) -> None:
    assert ts.changelog_url(url) == "https://raw.githubusercontent.com/o/r/main/.template/CHANGELOG.md"


def test_changelog_url_rejects_other_hosts() -> None:
    with pytest.raises(ValueError):
        ts.changelog_url("https://example.com/o/r")


def _project(tmp_path: Path, version: int) -> Path:
    marker = tmp_path / ".template-version"
    marker.write_text(f"# comment\nversion: {version}\nurl: https://github.com/o/r\nowned: scripts/sbatch src/utils\n")
    (tmp_path / "CHANGELOG.md").write_text(CHANGELOG)
    return marker


def test_behind_lists_only_newer_versions_and_the_diff_hint(tmp_path, monkeypatch, capsys) -> None:
    marker = _project(tmp_path, 1)
    monkeypatch.setenv("TEMPLATE_CHANGELOG_URL", (tmp_path / "CHANGELOG.md").as_uri())
    assert ts.main(["--marker", str(marker)]) == 0
    out = capsys.readouterr().out
    assert "template v1" in out and "latest: v3" in out and "behind by 2" in out
    assert "## v3" in out and "## v2" in out and "추가: A" not in out  # v1 is ours already
    assert "git diff template-v1 template-v3 -- scripts/sbatch src/utils" in out


def test_up_to_date(tmp_path, monkeypatch, capsys) -> None:
    marker = _project(tmp_path, 3)
    monkeypatch.setenv("TEMPLATE_CHANGELOG_URL", (tmp_path / "CHANGELOG.md").as_uri())
    assert ts.main(["--marker", str(marker)]) == 0
    assert "up to date" in capsys.readouterr().out


def test_mark_synced_rewrites_only_the_version_line(tmp_path, monkeypatch) -> None:
    marker = _project(tmp_path, 1)
    monkeypatch.setenv("TEMPLATE_CHANGELOG_URL", (tmp_path / "CHANGELOG.md").as_uri())
    assert ts.main(["--marker", str(marker), "--mark-synced"]) == 0
    text = marker.read_text()
    assert "version: 3" in text and "url: https://github.com/o/r" in text and "# comment" in text


def test_missing_marker_and_unreadable_changelog_fail_cleanly(tmp_path, monkeypatch, capsys) -> None:
    assert ts.main(["--marker", str(tmp_path / "nope")]) == 2
    marker = _project(tmp_path, 1)
    monkeypatch.setenv("TEMPLATE_CHANGELOG_URL", (tmp_path / "missing.md").as_uri())
    assert ts.main(["--marker", str(marker)]) == 1
    assert "could not read" in capsys.readouterr().err


# ---- the template repo's own bookkeeping (these files are deleted in a project, so skip there) ----
_HAS_TEMPLATE_DIR = (ROOT / ".template" / "CHANGELOG.md").is_file()


@pytest.mark.skipif(not _HAS_TEMPLATE_DIR, reason="a project created from the template has no .template/")
def test_template_version_matches_the_latest_changelog_entry() -> None:
    marker = ts.read_marker(ROOT / ts.MARKER)
    entries = ts.parse_changelog((ROOT / ".template" / "CHANGELOG.md").read_text())
    assert int(marker["version"]) == entries[0].version, "bump `.template-version` together with CHANGELOG.md"


@pytest.mark.skipif(not _HAS_TEMPLATE_DIR, reason="a project created from the template has no .template/")
def test_every_changelog_entry_says_what_downstream_has_to_do() -> None:
    for entry in ts.parse_changelog((ROOT / ".template" / "CHANGELOG.md").read_text()):
        assert "다운스트림에서 할 일" in entry.body and "영향 경로" in entry.body, f"v{entry.version}"


# ---- make rename ----
pytestmark_make = pytest.mark.skipif(
    platform.system() == "Windows" or shutil.which("make") is None, reason="needs make and a POSIX shell"
)


def _copy_project(tmp_path: Path, origin: str) -> Path:
    for name in ("Makefile", "README.md", "environment.yaml", ".env.example", ".template-version"):
        shutil.copy(ROOT / name, tmp_path / name)
    (tmp_path / ".template").mkdir()
    (tmp_path / ".template" / "MAINTAINING.md").write_text("x")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "remote", "add", "origin", origin], cwd=tmp_path, check=True)
    return tmp_path


@pytestmark_make
def test_rename_in_a_project_renames_everywhere_and_removes_the_template_folder(tmp_path) -> None:
    project = _copy_project(tmp_path, "https://github.com/someone/my-new-project.git")
    out = subprocess.run(["make", "rename", "NAME=my-new-project"], cwd=project, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert (project / "README.md").read_text().splitlines()[0] == "# my-new-project"
    assert "name: my-new-project" in (project / "environment.yaml").read_text()
    assert "PROJECT_NAME=my-new-project" in (project / ".env.example").read_text()
    assert not (project / ".template").exists()
    assert (project / ".template-version").exists()  # the project's own marker stays


@pytestmark_make
def test_rename_refuses_inside_the_template_repo_itself(tmp_path) -> None:
    project = _copy_project(tmp_path, "git@github.com:enjoeyland/research-template.git")  # ssh form of the url in the marker
    before = (project / "README.md").read_text()
    out = subprocess.run(["make", "rename", "NAME=oops"], cwd=project, capture_output=True, text=True)
    assert out.returncode != 0 and "template repo itself" in out.stdout + out.stderr
    assert (project / "README.md").read_text() == before and (project / ".template").exists()
    forced = subprocess.run(["make", "rename", "NAME=ok", "FORCE=1"], cwd=project, capture_output=True, text=True)
    assert forced.returncode == 0 and not (project / ".template").exists()


@pytestmark_make
def test_rename_rejects_bad_names(tmp_path) -> None:
    project = _copy_project(tmp_path, "https://github.com/someone/x.git")
    out = subprocess.run(["make", "rename", "NAME=bad name"], cwd=project, capture_output=True, text=True)
    assert out.returncode != 0 and (project / ".template").exists()
