"""Detect if a chapter is narrative content that should be segmented."""

import json
from pathlib import Path


def is_narrative_chapter(chapter_file: str | Path) -> tuple[bool, str]:
    """Check if a chapter is narrative content that should be segmented.

    Args:
        chapter_file: Path to chapter JSON file

    Returns:
        (is_narrative, reason) tuple
    """
    chapter_file = Path(chapter_file)
    with open(chapter_file, encoding="utf-8") as f:
        data = json.load(f)

    content = data.get("content", "")
    title = data.get("title", "")
    path = data.get("path", "").lower()

    # Non-narrative indicators
    non_narrative_patterns = [
        "caution",  # 警告页
        "copyright",  # 版权页
        "toc",  # 目录
        "contents",  # 目录
        "colophon",  # 版权页
        "navigation",  # 导航页
        "奥付",  # 日文版权页
        "目次",  # 日文目录
    ]

    # Check path
    for pattern in non_narrative_patterns:
        if pattern in path:
            return False, f"Non-narrative path: {path}"

    # Check title
    for pattern in non_narrative_patterns:
        if pattern in title.lower():
            return False, f"Non-narrative title: {title}"

    # Check content length (very short = likely non-narrative)
    if len(content) < 200:
        return False, f"Content too short ({len(content)} chars): likely metadata"

    # Check for narrative indicators
    narrative_indicators = ["。", "──", "※", "「", "』"]
    indicator_count = sum(1 for c in narrative_indicators if c in content)

    if indicator_count < 3:
        return False, f"No narrative indicators (found {indicator_count}/5)"

    # Check if it's mostly punctuation/symbols
    alpha_chars = sum(1 for c in content if c.isalpha())
    if alpha_chars / len(content) < 0.3:
        return False, f"Too few alphabetic characters ({alpha_chars}/{len(content)})"

    return True, "Narrative content"


def check_chapter(chapter_index: int, base_dir: str | Path) -> dict:
    """Check a single chapter and return status.

    Args:
        chapter_index: Chapter number
        base_dir: Base directory

    Returns:
        Status dict with is_narrative and reason
    """
    base_dir = Path(base_dir)
    chapter_file = base_dir / f"chapters/chapter_{chapter_index:03d}.json"

    if not chapter_file.exists():
        return {
            "chapter_index": chapter_index,
            "is_narrative": None,
            "reason": "File not found",
        }

    is_narrative, reason = is_narrative_chapter(chapter_file)
    return {
        "chapter_index": chapter_index,
        "is_narrative": is_narrative,
        "reason": reason,
    }


def check_all_chapters(base_dir: str | Path, start: int = 0, end: int = 16) -> list:
    """Check all chapters in range.

    Args:
        base_dir: Base directory
        start: Start chapter
        end: End chapter (exclusive)

    Returns:
        List of status dicts
    """
    base_dir = Path(base_dir)
    results = []

    for i in range(start, end):
        results.append(check_chapter(i, base_dir))

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Check if chapters are narrative")
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

    results = check_all_chapters(args.base, args.start, args.end)

    for r in results:
        status = "✓ NARRATIVE" if r["is_narrative"] else "✗ SKIP"
        print(f"ch{r['chapter_index']:03d}: {status} - {r['reason']}")
