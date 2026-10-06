"""How far is this project behind the template it was created from?  (`make template-status`)

Reads `.template-version` (the version this project was created from / last synced with, the template's URL and the paths the
template owns), fetches the template's `.template/CHANGELOG.md` from GitHub and lists the versions newer than ours with what
to do about each. Standard library only (no torch / lightning import), so it runs in any Python.

    python src/template_status.py                 # compare with the template
    python src/template_status.py --mark-synced   # after applying the changes: record the latest version in .template-version

TEMPLATE_CHANGELOG_URL overrides the changelog location (any URL urllib opens, e.g. file:///path/CHANGELOG.md).
"""

import argparse
import os
import re
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

MARKER = ".template-version"
_ENTRY = re.compile(r"^## v(\d+) \((\d{4}-\d{2}-\d{2})\)\s*(?:[—-]\s*(.*))?$")


@dataclass
class Entry:
    version: int
    date: str
    title: str
    body: str


def read_marker(path: Path) -> Dict[str, str]:
    """`key: value` lines; `#` starts a comment."""
    values: Dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip() if line.lstrip().startswith("#") else line.strip()
        if ":" in line and line:
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip()
    return values


def changelog_url(repo_url: str) -> str:
    """https://github.com/<owner>/<repo>[.git]  ->  the raw URL of `.template/CHANGELOG.md` on its main branch."""
    m = re.match(r"^(?:https://|git@)github\.com[/:]([^/]+)/([^/]+?)(?:\.git)?/?$", repo_url.strip())
    if not m:
        raise ValueError(f"not a GitHub repository URL: {repo_url!r} (set TEMPLATE_CHANGELOG_URL to override)")
    return f"https://raw.githubusercontent.com/{m.group(1)}/{m.group(2)}/main/.template/CHANGELOG.md"


def parse_changelog(text: str) -> List[Entry]:
    entries: List[Entry] = []
    current: Optional[Entry] = None
    for line in text.splitlines():
        m = _ENTRY.match(line)
        if m:
            current = Entry(int(m.group(1)), m.group(2), (m.group(3) or "").strip(), "")
            entries.append(current)
        elif current is not None:
            current.body += line + "\n"
    for entry in entries:
        entry.body = entry.body.strip("\n")
    return sorted(entries, key=lambda e: e.version, reverse=True)


def newer_than(entries: List[Entry], mine: int) -> List[Entry]:
    return [e for e in entries if e.version > mine]


def mark_synced(path: Path, version: int) -> None:
    lines = path.read_text().splitlines()
    out = [f"version: {version}" if re.match(r"^version\s*:", ln) else ln for ln in lines]
    path.write_text("\n".join(out) + "\n")


def fetch(url: str, timeout: float = 15.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 (a fixed https/file URL we build)
        return response.read().decode("utf-8")


def report(marker: Dict[str, str], entries: List[Entry]) -> str:
    mine = int(marker["version"])
    latest = entries[0].version if entries else mine
    newer = newer_than(entries, mine)
    lines = []
    if not newer:
        lines.append(f"up to date: this project is on template v{mine} (latest v{latest}).")
        return "\n".join(lines)
    lines.append(f"this project: template v{mine}    latest: v{latest}    behind by {len(newer)} version(s)")
    lines.append("")
    for e in newer:
        lines.append(f"## v{e.version} ({e.date}){' — ' + e.title if e.title else ''}")
        lines.append(e.body)
        lines.append("")
    owned = marker.get("owned", "").split()
    url = marker.get("url", "<template-url>")
    lines.append("To see the file-level differences of the template-owned paths:")
    lines.append(f"  git fetch {url} 'refs/tags/template-v*:refs/tags/template-v*'")
    lines.append(f"  git diff template-v{mine} template-v{latest} -- {' '.join(owned) if owned else '<paths>'}")
    lines.append("After applying what is relevant:  make template-mark-synced")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mark-synced", action="store_true", help="write the template's latest version into .template-version")
    parser.add_argument("--marker", default=MARKER, help=f"path of the version marker (default: {MARKER})")
    args = parser.parse_args(argv)

    path = Path(args.marker)
    if not path.is_file():
        print(f"error: {path} not found. Copy the template's {MARKER} and set `version:` to the template version this project "
              "was created from.", file=sys.stderr)
        return 2
    marker = read_marker(path)
    if "version" not in marker or not marker["version"].isdigit():
        print(f"error: {path} needs a `version: <integer>` line", file=sys.stderr)
        return 2

    try:
        url = os.environ.get("TEMPLATE_CHANGELOG_URL") or changelog_url(marker.get("url", ""))
        entries = parse_changelog(fetch(url))
    except Exception as exc:  # noqa: BLE001  (network, URL, ...: report and fail without a traceback)
        print(f"error: could not read the template's changelog: {exc}", file=sys.stderr)
        return 1
    if not entries:
        print("error: no `## vN (YYYY-MM-DD)` entries found in the changelog", file=sys.stderr)
        return 1

    if args.mark_synced:
        mark_synced(path, entries[0].version)
        print(f"{path}: version -> {entries[0].version}")
        return 0
    print(report(marker, entries))
    return 0


if __name__ == "__main__":
    sys.exit(main())
