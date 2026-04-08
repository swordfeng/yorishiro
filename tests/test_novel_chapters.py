from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from yorishiro.novel.chapter_extraction import (
    Chapter,
    MarkdownHeading,
    build_text_chapter,
    choose_markdown_chapter_level,
    detect_chapter_heading,
    extract_chapters,
    extract_text_from_html,
    extract_title_from_html,
    infer_title_from_text,
    looks_like_chapter_title,
    normalize_chapter_split_config,
    parse_markdown_headings,
    parse_text_source,
    split_markdown_chapters,
    split_markdown_source,
    split_text_chapters,
    strip_markdown_frontmatter,
)
from yorishiro.project import Project
from yorishiro.tasks.registry import ModelRegistry
from yorishiro.tasks.novel.chapters import NovelChaptersStep, NovelChaptersTask


class SplitTextChaptersTests(unittest.TestCase):
    def test_markdown_headings_split_into_chapters(self) -> None:
        text = """# Chapter 1

Alpha scene.

## Chapter 2

Beta scene.
"""
        chapters = split_text_chapters(text)

        self.assertEqual(len(chapters), 2)
        self.assertEqual(chapters[0]["title"], "Chapter 1")
        self.assertEqual(chapters[0]["content"], "Alpha scene.")
        self.assertEqual(chapters[0]["path"], "line:1")
        self.assertEqual(chapters[1]["title"], "Chapter 2")
        self.assertEqual(chapters[1]["content"], "Beta scene.")
        self.assertEqual(chapters[1]["path"], "line:5")

    def test_plain_text_markers_split_into_chapters(self) -> None:
        text = """第1章 始まり

最初の場面。

第2章 転換

次の場面。
"""
        chapters = split_text_chapters(text)

        self.assertEqual(len(chapters), 2)
        self.assertEqual(chapters[0]["title"], "第1章 始まり")
        self.assertIn("最初の場面。", chapters[0]["content"])
        self.assertEqual(chapters[1]["title"], "第2章 転換")
        self.assertIn("次の場面。", chapters[1]["content"])

    def test_missing_headings_falls_back_to_single_chapter(self) -> None:
        text = "A long body of prose without explicit chapter markers.\nStill the same unit."
        chapters = split_text_chapters(text)

        self.assertEqual(len(chapters), 1)
        self.assertIn("A long body of prose", chapters[0]["title"])
        self.assertEqual(chapters[0]["content"], text)
        self.assertEqual(chapters[0]["path"], "")

    def test_empty_text_returns_placeholder_chapter(self) -> None:
        chapters = split_text_chapters(" \n\r\n ")

        self.assertEqual(
            chapters,
            [{"title": "Chapter 1", "content": "", "path": ""}],
        )

    def test_preamble_before_first_heading_is_preserved(self) -> None:
        text = """Book title
Intro paragraph.

# Chapter 1

Real chapter text.
"""
        chapters = split_text_chapters(text)

        self.assertEqual(len(chapters), 2)
        self.assertEqual(chapters[0]["content"], "Book title\nIntro paragraph.")
        self.assertEqual(chapters[0]["path"], "line:1")
        self.assertEqual(chapters[1]["title"], "Chapter 1")

    def test_heading_with_no_body_is_dropped_and_falls_back_to_full_text(self) -> None:
        chapters = split_text_chapters("# Chapter 1")

        self.assertEqual(len(chapters), 1)
        self.assertEqual(chapters[0]["title"], "# Chapter 1")
        self.assertEqual(chapters[0]["content"], "# Chapter 1")


class HeadingHeuristicsTests(unittest.TestCase):
    def test_detect_chapter_heading_accepts_markdown_heading(self) -> None:
        self.assertEqual(detect_chapter_heading("## Chapter 12: Reunion", 4), "Chapter 12: Reunion")

    def test_detect_chapter_heading_rejects_non_chapter_markdown_heading(self) -> None:
        self.assertIsNone(detect_chapter_heading("## Character Notes", 2))

    def test_detect_chapter_heading_rejects_very_long_first_line(self) -> None:
        self.assertIsNone(detect_chapter_heading("a" * 121, 0))

    def test_looks_like_chapter_title_handles_common_patterns(self) -> None:
        self.assertTrue(looks_like_chapter_title("Chapter 7 - Arrival"))
        self.assertTrue(looks_like_chapter_title("序章"))
        self.assertTrue(looks_like_chapter_title("第十話 再会"))
        self.assertTrue(looks_like_chapter_title("Interlude"))
        self.assertFalse(looks_like_chapter_title("Character Notes"))
        self.assertFalse(looks_like_chapter_title("x" * 121))

    def test_looks_like_chapter_title_honors_custom_matcher(self) -> None:
        self.assertTrue(looks_like_chapter_title("Scene 3", matcher=r"^Scene \d+$"))
        self.assertFalse(looks_like_chapter_title("Chapter 3", matcher=r"^Scene \d+$"))


class MarkdownHeadingSplitTests(unittest.TestCase):
    def test_parse_markdown_headings_reads_levels_and_lines(self) -> None:
        headings = parse_markdown_headings(["# Title", "", "## One", "Body", "#### Deep"])

        self.assertEqual(
            headings,
            [
                MarkdownHeading(level=1, text="Title", line_index=0),
                MarkdownHeading(level=2, text="One", line_index=2),
                MarkdownHeading(level=4, text="Deep", line_index=4),
            ],
        )

    def test_choose_markdown_chapter_level_prefers_repeated_sibling_level(self) -> None:
        headings = [
            MarkdownHeading(level=1, text="Book", line_index=0),
            MarkdownHeading(level=2, text="One", line_index=2),
            MarkdownHeading(level=2, text="Two", line_index=8),
            MarkdownHeading(level=4, text="Detail", line_index=14),
        ]

        self.assertEqual(choose_markdown_chapter_level(headings, {1}), 2)

    def test_split_markdown_chapters_infers_h2_under_title(self) -> None:
        text = """# 竹取物語

## かぐや姫おひたち

第一章本文。

## つまどひ

第二章本文。
"""
        chapters = split_markdown_chapters(text)

        self.assertEqual(len(chapters), 2)
        self.assertEqual(chapters[0]["title"], "かぐや姫おひたち")
        self.assertEqual(chapters[0]["content"], "第一章本文。")
        self.assertEqual(chapters[0]["path"], "line:3")
        self.assertEqual(chapters[1]["title"], "つまどひ")

    def test_split_markdown_chapters_can_use_h4_when_configured(self) -> None:
        text = """# Book

## Part One

#### Scene A

Alpha

#### Scene B

Beta
"""
        chapters = split_markdown_chapters(text, levels=[4], matcher=r"^Scene ")

        self.assertEqual(len(chapters), 2)
        self.assertEqual(chapters[0]["title"], "Scene A")
        self.assertEqual(chapters[1]["title"], "Scene B")

    def test_split_markdown_chapters_honors_matcher_and_exclude_matcher(self) -> None:
        text = """# Book

## TOC

skip

## Chapter Alpha

Alpha

## Notes

skip

## Chapter Beta

Beta
"""
        chapters = split_markdown_chapters(
            text,
            matcher=r"^Chapter ",
            exclude_matcher=r"Beta$",
        )

        self.assertEqual(len(chapters), 1)
        self.assertEqual(chapters[0]["title"], "Chapter Alpha")

    def test_split_markdown_source_can_fallback_to_text_markers(self) -> None:
        text = """序章

本文

終章

結末
"""
        chapters = split_markdown_source(text, normalize_chapter_split_config(None))

        self.assertEqual(len(chapters), 2)
        self.assertEqual(chapters[0]["title"], "序章")

    def test_split_markdown_source_mode_none_keeps_single_chapter(self) -> None:
        text = """# Title

## One

Alpha
"""
        cfg = normalize_chapter_split_config({"chapter_split": {"mode": "none"}})
        chapters = split_markdown_source(text, cfg)

        self.assertEqual(len(chapters), 1)
        self.assertEqual(chapters[0]["content"], "# Title\n\n## One\n\nAlpha")


class MarkdownUtilityTests(unittest.TestCase):
    def test_strip_markdown_frontmatter_removes_leading_block_only(self) -> None:
        text = "---\ntitle: Demo\nlayout: novel\n---\n# Chapter 1\n\nBody"
        self.assertEqual(strip_markdown_frontmatter(text), "# Chapter 1\n\nBody")

    def test_strip_markdown_frontmatter_leaves_non_frontmatter_text_unchanged(self) -> None:
        text = "Chapter marker\n---\nThis is content."
        self.assertEqual(strip_markdown_frontmatter(text), text)

    def test_build_text_chapter_drops_markdown_heading_line(self) -> None:
        title, content = build_text_chapter(["# Chapter 1", "", "Body"], 0, 3, "Chapter 1")
        self.assertEqual(title, "Chapter 1")
        self.assertEqual(content, "Body")

    def test_build_text_chapter_drops_plain_heading_line(self) -> None:
        title, content = build_text_chapter(["第1章 始まり", "", "本文"], 0, 3, "第1章 始まり")
        self.assertEqual(title, "第1章 始まり")
        self.assertEqual(content, "本文")

    def test_infer_title_from_text_uses_first_non_blank_line(self) -> None:
        lines = ["", "  ", "First useful line", "Second line"]
        self.assertEqual(infer_title_from_text(lines), "First useful line")


class ParseTextSourceTests(unittest.TestCase):
    def test_parse_text_source_strips_bom_and_markdown_frontmatter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "novel.md"
            source.write_text(
                "\ufeff---\ntitle: Demo\n---\n# Chapter 1\n\nAlpha.\n",
                encoding="utf-8",
            )

            chapters = list(parse_text_source(source))

            self.assertEqual(len(chapters), 1)
            self.assertEqual(chapters[0].title, "Chapter 1")
            self.assertEqual(chapters[0].content, "Alpha.")
            self.assertEqual(chapters[0].path, "line:1")

    def test_parse_text_source_uses_fallback_path_for_unsplit_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "novel.txt"
            source.write_text("Standalone prose body.", encoding="utf-8")

            chapters = list(parse_text_source(source))

            self.assertEqual(len(chapters), 1)
            self.assertEqual(chapters[0].title, "Standalone prose body.")
            self.assertEqual(chapters[0].path, "text-chapter-0")

    def test_parse_text_source_uses_markdown_structure_for_bamboo_style_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "novel.md"
            source.write_text(
                """# 竹取物語

## かぐや姫おひたち

最初の章。

## つまどひ

次の章。
""",
                encoding="utf-8",
            )

            chapters = list(parse_text_source(source))

            self.assertEqual([chapter.title for chapter in chapters], ["かぐや姫おひたち", "つまどひ"])

    def test_parse_text_source_honors_chapter_split_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "novel.md"
            source.write_text(
                """# Book

## Intro

skip

#### Scene 1

Alpha

#### Scene 2

Beta
""",
                encoding="utf-8",
            )

            chapters = list(
                parse_text_source(
                    source,
                    source_config={
                        "chapter_split": {
                            "mode": "markdown_headers",
                            "markdown": {
                                "levels": [4],
                                "matcher": r"^Scene ",
                                "title_levels": [1, 2],
                            },
                        }
                    },
                )
            )

            self.assertEqual([chapter.title for chapter in chapters], ["Scene 1", "Scene 2"])


class ExtractTextFromHtmlTests(unittest.TestCase):
    def test_extract_text_from_html_strips_scripts_styles_and_normalizes_breaks(self) -> None:
        html = """
        <html>
          <head><style>.x { color: red; }</style></head>
          <body>
            <script>alert(1)</script>
            <p>First   line<br/>next</p>
            <p>Second line</p>
          </body>
        </html>
        """
        text = extract_text_from_html(html)

        self.assertEqual(text, "First line\nnext\nSecond line")

    def test_extract_text_from_html_removes_furigana_by_default(self) -> None:
        html = "<ruby><rb>漢字</rb><rt>かんじ</rt></ruby>"
        self.assertEqual(extract_text_from_html(html), "漢字")

    def test_extract_text_from_html_can_keep_furigana(self) -> None:
        html = "<ruby>漢<rb>字</rb><rt>かんじ</rt></ruby>"
        self.assertEqual(extract_text_from_html(html, remove_furigana=False), "漢字")
        self.assertEqual(extract_text_from_html(html, remove_furigana=True), "字漢")


class ExtractTitleFromHtmlTests(unittest.TestCase):
    def test_extract_title_from_html_prefers_h1_then_title_then_strong(self) -> None:
        self.assertEqual(extract_title_from_html("<h1>Visible Heading</h1><title>Fallback</title>"), "Visible Heading")
        self.assertEqual(extract_title_from_html("<title>Only Title</title>"), "Only Title")
        self.assertEqual(extract_title_from_html("<strong>Short Strong Title</strong>"), "Short Strong Title")

    def test_extract_title_from_html_ignores_overlong_strong_text(self) -> None:
        html = f"<strong>{'x' * 101}</strong>"
        self.assertIsNone(extract_title_from_html(html))


class ExtractChaptersTests(unittest.TestCase):
    def test_extract_chapters_missing_file_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "missing.txt"
            output_dir = Path(tmp_dir) / "out"

            with self.assertRaises(FileNotFoundError):
                extract_chapters(source, output_dir)

    def test_extract_chapters_unsupported_suffix_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "novel.docx"
            output_dir = Path(tmp_dir) / "out"
            source.write_text("body", encoding="utf-8")

            with self.assertRaises(ValueError):
                extract_chapters(source, output_dir)

    def test_markdown_file_writes_yaml_frontmatter_chapters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = root / "novel.md"
            output_dir = root / "out"
            source.write_text(
                """---
title: Demo
---
# Chapter 1

Alpha.

# Chapter 2

Beta.
""",
                encoding="utf-8",
            )

            saved = extract_chapters(source, output_dir)

            self.assertEqual([path.name for path in saved], ["ch000.txt", "ch001.txt"])
            metadata, content = parse_saved_chapter(saved[0])
            self.assertEqual(metadata["index"], 0)
            self.assertEqual(metadata["title"], "Chapter 1")
            self.assertEqual(metadata["source_file"], "novel.md")
            self.assertEqual(metadata["path"], "line:1")
            self.assertEqual(metadata["length"], len(content))
            self.assertEqual(content, "Alpha.")

    def test_extract_chapters_dispatches_epub_to_parse_epub(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = root / "novel.epub"
            output_dir = root / "out"
            source.write_bytes(b"fake epub")

            fake_chapters = [
                Chapter(index=0, title="Start", content="Alpha", path="OPS/ch1.xhtml"),
                Chapter(index=1, title="End", content="Beta", path="OPS/ch2.xhtml"),
            ]

            with patch("yorishiro.novel.chapter_extraction.parse_epub", return_value=iter(fake_chapters)) as mock_parse:
                saved = extract_chapters(source, output_dir)

            mock_parse.assert_called_once_with(source, remove_furigana=True)
            self.assertEqual([path.name for path in saved], ["ch000.txt", "ch001.txt"])
            metadata, content = parse_saved_chapter(saved[1])
            self.assertEqual(metadata["title"], "End")
            self.assertEqual(metadata["path"], "OPS/ch2.xhtml")
            self.assertEqual(content, "Beta")

    def test_extract_chapters_passes_source_config_to_text_parser(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = root / "novel.md"
            output_dir = root / "out"
            source.write_text("body", encoding="utf-8")

            with patch("yorishiro.novel.chapter_extraction.parse_text_source", return_value=iter([])) as mock_parse:
                extract_chapters(source, output_dir, source_config={"chapter_split": {"mode": "none"}})

            mock_parse.assert_called_once_with(source, source_config={"chapter_split": {"mode": "none"}})


class NovelChaptersTaskTests(unittest.TestCase):
    def test_output_paths_and_completion_marker_use_existing_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = root / "novel.txt"
            output_dir = root / "chapters"
            output_dir.mkdir()
            source.write_text("Body", encoding="utf-8")
            (output_dir / "ch001.txt").write_text("", encoding="utf-8")
            (output_dir / "ch010.txt").write_text("", encoding="utf-8")

            task = NovelChaptersTask(source, output_dir)

            self.assertEqual(task.input_paths(), [source])
            self.assertEqual(task.output_paths(), [output_dir / "ch001.txt", output_dir / "ch010.txt"])
            self.assertEqual(task.completion_marker(), output_dir / "ch010.txt")

    def test_output_paths_fallback_when_no_existing_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = root / "novel.txt"
            output_dir = root / "chapters"
            source.write_text("Body", encoding="utf-8")

            task = NovelChaptersTask(source, output_dir)

            self.assertEqual(task.output_paths(), [output_dir / "ch000.txt"])
            self.assertEqual(task.completion_marker(), output_dir / "ch000.txt")


class NovelChaptersStepTests(unittest.TestCase):
    def test_step_builds_task_from_project_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            project_yaml = root / "project.yaml"
            raw_dir = root / "raw"
            raw_dir.mkdir()
            source_path = raw_dir / "novel.md"
            source_path.write_text("# Chapter 1\n\nAlpha", encoding="utf-8")
            project_yaml.write_text(
                """project:
  name: Demo
  code: demo
sources:
  - id: novel-src
    type: novel
    path: raw/novel.md
""",
                encoding="utf-8",
            )

            project = Project.load(root)
            step = NovelChaptersStep(project, "novel-src", registry=ModelRegistry(project))

            tasks = step.tasks()

            self.assertEqual(len(tasks), 1)
            self.assertIsInstance(tasks[0], NovelChaptersTask)
            self.assertEqual(tasks[0].input_paths(), [source_path])
            self.assertEqual(tasks[0].output_paths(), [root / "processed" / "novel-src" / "steps" / "chapters" / "ch000.txt"])

    def test_step_passes_source_config_to_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "raw").mkdir()
            (root / "raw" / "novel.md").write_text("# Book", encoding="utf-8")
            (root / "project.yaml").write_text(
                """project:
  name: Demo
  code: demo
sources:
  - id: novel-src
    type: novel
    path: raw/novel.md
    config:
      chapter_split:
        mode: markdown_headers
        markdown:
          levels: [2]
          matcher: ".*"
""",
                encoding="utf-8",
            )

            project = Project.load(root)
            task = NovelChaptersStep(project, "novel-src", registry=ModelRegistry(project)).tasks()[0]
            self.assertIsInstance(task, NovelChaptersTask)
            assert isinstance(task, NovelChaptersTask)

            self.assertEqual(
                task._source_config,
                {
                    "chapter_split": {
                        "mode": "markdown_headers",
                        "markdown": {"levels": [2], "matcher": ".*"},
                    }
                },
            )


def parse_saved_chapter(path: Path) -> tuple[dict, str]:
    text = path.read_text(encoding="utf-8")
    _, raw_meta, body = text.split("---", 2)
    return yaml.safe_load(raw_meta), body.lstrip("\n")


if __name__ == "__main__":
    unittest.main()
