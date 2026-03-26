"""Verify scene segmentation with offset checking."""

import json
from pathlib import Path


def verify_chapter_scenes(chapter_index: int, base_dir: str | Path) -> dict:
    """Verify scene segmentation for a chapter with offset checking.

    Args:
        chapter_index: Chapter number
        base_dir: Base directory containing chapters and scenes

    Returns:
        Verification report dict
    """
    base_dir = Path(base_dir)

    # Read original chapter
    chapter_file = base_dir / f"chapters/chapter_{chapter_index:03d}.json"
    if not chapter_file.exists():
        return {"chapter_index": chapter_index, "error": f"Chapter file not found"}

    with open(chapter_file, encoding="utf-8") as f:
        chapter_data = json.load(f)

    original_text = chapter_data.get("content", "")
    original_length = len(original_text)

    # Read scenes directory
    scenes_dir = base_dir / f"scenes/ch{chapter_index:03d}"
    if not scenes_dir.exists():
        return {"chapter_index": chapter_index, "error": f"S scenes directory not found"}

    # Read manifest
    manifest_file = scenes_dir / "scenes_manifest.json"
    if not manifest_file.exists():
        return {"chapter_index": chapter_index, "error": f"Manifest not found"}

    with open(manifest_file, encoding="utf-8") as f:
        manifest = json.load(f)

    # Check manifest has total_length
    manifest_total = manifest.get("total_length")
    if manifest_total is None:
        return {
            "chapter_index": chapter_index,
            "error": "Manifest missing total_length",
            "status": "ISSUE"
        }

    if manifest_total != original_length:
        return {
            "chapter_index": chapter_index,
            "error": f"total_length mismatch: manifest={manifest_total}, original={original_length}",
            "status": "ISSUE"
        }

    scenes = manifest.get("scenes", [])
    scene_count = len(scenes)

    # Read scene files and verify offsets
    offset_issues = []
    content_issues = []
    expected_offset = 0

    for i, scene_meta in enumerate(scenes):
        scene_file_path = scenes_dir / scene_meta.get("file", f"scene_{i:03d}.txt")
        if not scene_file_path.exists():
            offset_issues.append(f"scene_{i:03d}: file not found")
            continue

        with open(scene_file_path, encoding="utf-8") as f:
            scene_content = f.read()

        start_offset = scene_meta.get("start_offset")
        end_offset = scene_meta.get("end_offset")

        if start_offset is None or end_offset is None:
            offset_issues.append(f"scene_{i:03d}: missing offset")
            continue

        # Check offset continuity
        if start_offset != expected_offset:
            offset_issues.append(
                f"scene_{i:03d}: gap/dup at offset {start_offset}, expected {expected_offset}"
            )

        # Check content matches original at offset
        expected_content = original_text[start_offset:end_offset]
        if expected_content != scene_content:
            content_issues.append(
                f"scene_{i:03d}: content mismatch at offset {start_offset}-{end_offset}"
            )

        expected_offset = end_offset

    # Check final offset
    if expected_offset != original_length:
        offset_issues.append(
            f"Final offset {expected_offset} != original length {original_length}"
        )

    status = "OK" if not offset_issues and not content_issues else "ISSUE"

    return {
        "chapter_index": chapter_index,
        "chapter_title": chapter_data.get("title", ""),
        "original_length": original_length,
        "scene_count": scene_count,
        "offset_issues": offset_issues,
        "content_issues": content_issues,
        "status": status,
    }


def verify_all_chapters(base_dir: str | Path, start: int = 0, end: int = 16) -> list:
    """Verify all chapters in range."""
    base_dir = Path(base_dir)
    reports = []
    for i in range(start, end):
        report = verify_chapter_scenes(i, base_dir)
        reports.append(report)
    return reports


def print_report(report: dict) -> None:
    """Print a single report."""
    ch = report.get("chapter_index", "?")
    if "error" in report:
        print(f"ch{ch:03d}: ERROR - {report['error']}")
        return

    status = "✓" if report["status"] == "OK" else "✗"
    print(f"\n{'='*60}")
    print(f"Chapter {ch}: {report['chapter_title']}")
    print(f"  Status: {status} {report['status']}")
    print(f"  Scene count: {report['scene_count']}")
    print(f"  Original length: {report['original_length']:,}")

    if report.get("offset_issues"):
        print(f"  ⚠️  Offset issues:")
        for issue in report["offset_issues"]:
            print(f"      - {issue}")

    if report.get("content_issues"):
        print(f"  ⚠️  Content issues:")
        for issue in report["content_issues"]:
            print(f"      - {issue}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Verify scene segmentation with offsets")
    parser.add_argument(
        "--base",
        default="material/processed/novel/CPK",
        help="Base directory",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start chapter",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=16,
        help="End chapter (exclusive)",
    )

    args = parser.parse_args()

    reports = verify_all_chapters(args.base, args.start, args.end)

    issues = 0
    for report in reports:
        print_report(report)
        if report.get("status") == "ISSUE":
            issues += 1

    print(f"\n{'='*60}")
    print(f"Total chapters checked: {len(reports)}")
    print(f"Issues found: {issues}")
