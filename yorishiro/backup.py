"""Automatic snapshot-based backup for project outputs.

Pure stdlib — no external dependencies.

Files unchanged since the last snapshot are stored as hard links rather than
copies, keeping disk usage proportional to the number of distinct versions.

Retention policy: every snapshot in the last 4 hours, one per hour for the
last 24 hours, one per day beyond that.

Usage:
    # Programmatic
    backup = ProjectBackup(project_dir)
    backup.snapshot("chapters")  # snapshot entire project

    # CLI
    uv run python -m yorishiro.backup --project <dir> --label manual
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path


_BACKUPS_DIR = ".yorishiro_backups"
_EXCLUDE_DIRS = {_BACKUPS_DIR, ".git", "__pycache__", "node_modules"}


class ProjectBackup:
    def __init__(self, project_dir: Path):
        self.project_dir = project_dir
        self.backups_dir = project_dir / _BACKUPS_DIR

    def snapshot(self, label: str = "") -> Path | None:
        """Snapshot entire project directory into a timestamped backup directory.

        Returns the backup path, or None if project directory is empty.
        Call after a write operation completes.
        Unchanged files are stored as hard links to save disk space.

        Excludes: .yorishiro_backups, .git, __pycache__, node_modules
        """
        prev_snapshot = self._last_snapshot()
        dest = self._make_dest(label)

        for src_file in self.project_dir.rglob("*"):
            if not src_file.is_file():
                continue
            if self._should_exclude(src_file):
                continue
            rel = src_file.relative_to(self.project_dir)
            self._hardlink_or_copy(src_file, dest / rel, prev_snapshot, rel)

        self._prune()
        return dest

    def _should_exclude(self, path: Path) -> bool:
        """Check if file should be excluded from backup."""
        for part in path.parts:
            if part in _EXCLUDE_DIRS:
                return True
        return False

    def _hardlink_or_copy(
        self, src: Path, dest: Path, prev_snapshot: Path | None, rel: Path
    ) -> None:
        """Write src to dest: hard link from previous snapshot if content matches, else copy."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        if prev_snapshot:
            prev_file = prev_snapshot / rel
            if prev_file.exists() and self._same_content(src, prev_file):
                try:
                    os.link(prev_file, dest)
                    return
                except OSError:
                    pass  # hard link unsupported; fall through to copy
        shutil.copy2(src, dest)

    def _same_content(self, a: Path, b: Path) -> bool:
        sa, sb = a.stat(), b.stat()
        if sa.st_dev == sb.st_dev and sa.st_ino == sb.st_ino:
            return True   # already the same inode
        if sa.st_size != sb.st_size:
            return False
        return sa.st_mtime == sb.st_mtime  # same size + same mtime → likely unchanged

    def _last_snapshot(self) -> Path | None:
        """Return most recent snapshot dir, or None if none exist."""
        if not self.backups_dir.exists():
            return None
        snapshots = sorted(
            self.backups_dir.iterdir(),
            key=lambda p: p.name,
            reverse=True,
        )
        for p in snapshots:
            if p.is_dir():
                return p
        return None

    def _make_dest(self, label: str) -> Path:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        safe_label = label.replace("/", "_").replace("\\", "_")
        # Include PID to prevent two concurrent processes from colliding on a
        # same-second same-label snapshot (mkdir exist_ok=True would silently
        # merge their writes into one corrupted snapshot dir without the PID).
        dest = self.backups_dir / f"{ts}_{safe_label}_{os.getpid()}"
        dest.mkdir(parents=True, exist_ok=True)
        return dest

    def _prune(self) -> None:
        """Tiered retention: all in last 4 h, one/hour for last 24 h, one/day beyond."""
        if not self.backups_dir.exists():
            return
        now = datetime.now()
        cutoff_4h = now - timedelta(hours=4)
        cutoff_24h = now - timedelta(hours=24)

        # newest-first so the first entry in each bucket is the one we keep
        snapshots = sorted(
            (p for p in self.backups_dir.iterdir() if p.is_dir()),
            key=lambda p: p.name,
            reverse=True,
        )

        keep: set[Path] = set()
        seen_hours: set[tuple] = set()
        seen_days: set[tuple] = set()

        for snap in snapshots:
            try:
                ts = datetime.strptime(snap.name[:15], "%Y%m%d-%H%M%S")
            except ValueError:
                keep.add(snap)   # unparseable name — don't touch
                continue

            if ts >= cutoff_4h:
                keep.add(snap)   # keep every snapshot in the last 4 hours
            elif ts >= cutoff_24h:
                bucket = (ts.year, ts.month, ts.day, ts.hour)
                if bucket not in seen_hours:
                    seen_hours.add(bucket)
                    keep.add(snap)   # keep newest of each clock-hour
            else:
                bucket = (ts.year, ts.month, ts.day)
                if bucket not in seen_days:
                    seen_days.add(bucket)
                    keep.add(snap)   # keep newest of each calendar-day

        for snap in snapshots:
            if snap not in keep:
                shutil.rmtree(snap, ignore_errors=True)


def main() -> None:
    """CLI entry point for manual backup."""
    parser = argparse.ArgumentParser(
        description="Create a backup snapshot of the entire project.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  uv run python -m yorishiro.backup --project projects/CPK\n"
            "  uv run python -m yorishiro.backup --project projects/CPK --label manual\n"
        ),
    )

    parser.add_argument(
        "--project",
        type=Path,
        default=None,
        help="Project directory (default: current directory)",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="manual",
        help="Label for the backup snapshot (default: manual)",
    )
    args = parser.parse_args()

    project_path = args.project if args.project else Path.cwd()

    try:
        from yorishiro.project import Project
        project = Project.load(project_path)
    except FileNotFoundError:
        print(f"Error: project.yaml not found at {project_path}", file=sys.stderr)
        sys.exit(1)

    backup = ProjectBackup(project.root)
    backup_path = backup.snapshot(args.label)

    if backup_path:
        print(f"Created backup: {backup_path}")
    else:
        print("No files to backup.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()