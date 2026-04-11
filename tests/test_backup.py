from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from yorishiro.backup import ProjectBackup


class ProjectBackupTests(unittest.TestCase):
    def test_snapshot_hardlinks_rewritten_same_content_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            target = root / "artifact.bin"
            target.write_bytes(b"hello world")

            backup = ProjectBackup(root)
            snap1 = backup.snapshot("s1")
            self.assertIsNotNone(snap1)
            assert snap1 is not None

            # Ensure mtime changes while content stays identical.
            time.sleep(1.1)
            target.write_bytes(b"hello world")

            snap2 = backup.snapshot("s2")
            self.assertIsNotNone(snap2)
            assert snap2 is not None

            p1 = snap1 / "artifact.bin"
            p2 = snap2 / "artifact.bin"
            st1 = p1.stat()
            st2 = p2.stat()

            self.assertEqual(st1.st_ino, st2.st_ino)
            self.assertGreaterEqual(st1.st_nlink, 2)

    def test_snapshot_copies_when_content_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            target = root / "artifact.bin"
            target.write_bytes(b"v1")

            backup = ProjectBackup(root)
            snap1 = backup.snapshot("s1")
            self.assertIsNotNone(snap1)
            assert snap1 is not None

            time.sleep(1.1)
            target.write_bytes(b"v2")

            snap2 = backup.snapshot("s2")
            self.assertIsNotNone(snap2)
            assert snap2 is not None

            p1 = snap1 / "artifact.bin"
            p2 = snap2 / "artifact.bin"
            st1 = p1.stat()
            st2 = p2.stat()

            self.assertNotEqual(st1.st_ino, st2.st_ino)
