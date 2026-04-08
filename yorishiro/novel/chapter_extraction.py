"""Chapter extraction and splitting logic for novel sources."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import yaml


TEXT_SOURCE_SUFFIXES = {".txt", ".md", ".markdown"}
CHAPTER_MARKER_RE = re.compile(
    r"^(?:"
    r"(?:chapter|chap\.?)\s+\d+[A-Za-z0-9\-.: ]*"
    r"|第\s*[0-9０-９一二三四五六七八九十百千]+(?:\s*[章話卷巻篇部節])(?:\s+.*)?"
    r"|(?:prologue|epilogue|interlude|afterword|foreword|preface|introduction)\b.*"
    r"|(?:序章|終章|番外編|外伝|幕間|後書き|あとがき|前書き|まえがき|序|プロローグ|エピローグ).*)$",
    re.IGNORECASE,
)


@dataclass
class Chapter:
    """Represents one extracted chapter."""

    index: int
    title: str
    content: str
    path: str = ""


@dataclass
class MarkdownHeading:
    """One Markdown heading line."""

    level: int
    text: str
    line_index: int


def extract_chapters(
    source_path: Path,
    output_dir: Path,
    remove_furigana: bool = True,
    source_config: dict[str, Any] | None = None,
) -> list[Path]:
    """Extract chapters from a supported source into YAML-frontmatter text files."""
    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")

    suffix = source_path.suffix.lower()
    if suffix == ".epub":
        chapters = list(parse_epub(source_path, remove_furigana=remove_furigana))
    elif suffix in TEXT_SOURCE_SUFFIXES:
        chapters = list(parse_text_source(source_path, source_config=source_config))
    else:
        raise ValueError(f"Unsupported chapter source format: {source_path.suffix or '<no suffix>'}")

    output_dir.mkdir(parents=True, exist_ok=True)
    digits = max(3, len(str(len(chapters))))
    saved_files: list[Path] = []

    for chapter in chapters:
        saved_files.append(save_chapter(chapter, output_dir, source_path.name, digits))

    return saved_files


def parse_epub(source_path: Path, remove_furigana: bool = True) -> Iterator[Chapter]:
    """Parse an EPUB file and yield text-bearing document items as chapters."""
    import ebooklib
    from ebooklib import epub

    book = epub.read_epub(str(source_path))
    chapter_count = 0

    for item in book.get_items():
        if item.get_type() != ebooklib.ITEM_DOCUMENT:
            continue

        html_content = item.get_content().decode("utf-8", errors="ignore")
        text = extract_text_from_html(html_content, remove_furigana=remove_furigana)
        if not text.strip():
            continue

        title = extract_title_from_html(html_content) or f"Chapter {chapter_count + 1}"
        item_path = getattr(item, "href", None) or getattr(item, "file_name", str(item.get_name()))
        yield Chapter(index=chapter_count, title=title, content=text, path=item_path)
        chapter_count += 1


def parse_text_source(source_path: Path, source_config: dict[str, Any] | None = None) -> Iterator[Chapter]:
    """Parse a plain text or Markdown source into one or more chapters."""
    text = source_path.read_text(encoding="utf-8").replace("\ufeff", "")
    split_cfg = normalize_chapter_split_config(source_config)

    if source_path.suffix.lower() in {".md", ".markdown"}:
        text = strip_markdown_frontmatter(text)
        chapters = split_markdown_source(text, split_cfg)
    else:
        chapters = split_text_chapters(text, matcher=split_cfg["text_matcher"])

    digits = max(1, len(str(len(chapters))))

    for index, chapter in enumerate(chapters):
        title = chapter["title"] or f"Chapter {index + 1}"
        path = chapter["path"] or f"text-chapter-{index:0{digits}d}"
        yield Chapter(index=index, title=title, content=chapter["content"], path=path)


def normalize_chapter_split_config(source_config: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize per-source chapter split config."""
    cfg = dict((source_config or {}).get("chapter_split", {}))
    markdown_cfg = dict(cfg.get("markdown", {}))
    text_cfg = dict(cfg.get("text", {}))
    return {
        "mode": cfg.get("mode", "auto"),
        "markdown_levels": [int(level) for level in markdown_cfg.get("levels", [])],
        "markdown_matcher": markdown_cfg.get("matcher"),
        "markdown_exclude_matcher": markdown_cfg.get("exclude_matcher"),
        "markdown_title_levels": [int(level) for level in markdown_cfg.get("title_levels", [1])],
        "text_matcher": text_cfg.get("matcher"),
    }


def strip_markdown_frontmatter(text: str) -> str:
    """Remove an optional leading Markdown frontmatter block."""
    if not text.startswith("---\n"):
        return text

    match = re.match(r"\A---\n.*?\n---\n?", text, flags=re.DOTALL)
    if match:
        return text[match.end():]
    return text


def split_markdown_source(text: str, split_config: dict[str, Any] | None = None) -> list[dict[str, str]]:
    """Split a Markdown source using heading structure and optional fallbacks."""
    cfg = split_config or normalize_chapter_split_config(None)
    mode = cfg["mode"]

    if mode not in {"auto", "markdown_headers", "text_markers", "none"}:
        raise ValueError(f"Unsupported chapter_split.mode: {mode}")

    if mode in {"auto", "markdown_headers"}:
        chapters = split_markdown_chapters(
            text,
            levels=cfg["markdown_levels"],
            matcher=cfg["markdown_matcher"],
            exclude_matcher=cfg["markdown_exclude_matcher"],
            title_levels=cfg["markdown_title_levels"],
        )
        if chapters:
            return chapters
        if mode == "markdown_headers":
            normalized = normalize_text(text)
            return [{"title": infer_title_from_text(normalized.split("\n")), "content": normalized, "path": ""}]

    if mode in {"auto", "text_markers"}:
        return split_text_chapters(text, matcher=cfg["text_matcher"])

    normalized = normalize_text(text)
    return [{"title": infer_title_from_text(normalized.split("\n")), "content": normalized, "path": ""}]


def parse_markdown_headings(lines: list[str]) -> list[MarkdownHeading]:
    """Parse ATX-style Markdown headings."""
    headings: list[MarkdownHeading] = []
    for idx, line in enumerate(lines):
        match = re.match(r"^(#{1,6})\s+(.*\S)\s*$", line.strip())
        if match:
            headings.append(MarkdownHeading(level=len(match.group(1)), text=match.group(2).strip(), line_index=idx))
    return headings


def split_markdown_chapters(
    text: str,
    *,
    levels: list[int] | None = None,
    matcher: str | None = None,
    exclude_matcher: str | None = None,
    title_levels: list[int] | None = None,
) -> list[dict[str, str]]:
    """Split a Markdown source by structural headings."""
    normalized = normalize_text(text)
    if not normalized:
        return []

    lines = normalized.split("\n")
    headings = parse_markdown_headings(lines)
    if not headings:
        return []

    heading_levels = set(levels or [])
    title_level_set = set(title_levels or [1])
    include_re = re.compile(matcher) if matcher else None
    exclude_re = re.compile(exclude_matcher) if exclude_matcher else None

    candidates = [
        heading
        for heading in headings
        if (not heading_levels or heading.level in heading_levels)
        and (include_re is None or include_re.search(heading.text))
        and (exclude_re is None or not exclude_re.search(heading.text))
    ]
    if not candidates:
        return []

    chosen_level = choose_markdown_chapter_level(candidates, title_level_set)
    chosen = [heading for heading in candidates if heading.level == chosen_level]
    if not chosen:
        return []

    chapters: list[dict[str, str]] = []
    total_lines = len(lines)
    for index, heading in enumerate(chosen):
        end_line = chosen[index + 1].line_index if index + 1 < len(chosen) else total_lines
        title, content = build_text_chapter(lines, heading.line_index, end_line, heading.text)
        if not content.strip():
            continue
        chapters.append(
            {
                "title": title,
                "content": content,
                "path": f"line:{heading.line_index + 1}",
            }
        )

    return chapters


def choose_markdown_chapter_level(headings: list[MarkdownHeading], title_levels: set[int]) -> int:
    """Choose the heading level most likely to represent chapter boundaries."""
    counts: dict[int, int] = {}
    for heading in headings:
        counts[heading.level] = counts.get(heading.level, 0) + 1

    repeated_levels = [level for level, count in counts.items() if count >= 2]
    if repeated_levels:
        return min(repeated_levels, key=lambda level: (-counts[level], level))

    if len(headings) >= 2:
        first_level = headings[0].level
        later_levels = [heading.level for heading in headings[1:] if heading.level not in title_levels or first_level == heading.level]
        if later_levels:
            later_counts: dict[int, int] = {}
            for level in later_levels:
                later_counts[level] = later_counts.get(level, 0) + 1
            return min(later_counts, key=lambda level: (-later_counts[level], level))

    return headings[0].level


def normalize_text(text: str) -> str:
    """Normalize newline conventions and strip outer whitespace."""
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def split_text_chapters(text: str, matcher: str | None = None) -> list[dict[str, str]]:
    """Split plain text source by chapter marker lines."""
    normalized = normalize_text(text)
    if not normalized:
        return [{"title": "Chapter 1", "content": "", "path": ""}]

    lines = normalized.split("\n")
    boundaries: list[tuple[int, str]] = []

    for idx, line in enumerate(lines):
        heading = detect_chapter_heading(line, idx, matcher=matcher)
        if heading is not None:
            boundaries.append((idx, heading))

    if not boundaries:
        return [{"title": infer_title_from_text(lines), "content": normalized, "path": ""}]

    chapters: list[dict[str, str]] = []
    if boundaries[0][0] > 0:
        preamble = "\n".join(lines[:boundaries[0][0]]).strip()
        if preamble:
            chapters.append(
                {
                    "title": infer_title_from_text(lines[:boundaries[0][0]]),
                    "content": preamble,
                    "path": "line:1",
                }
            )

    total_lines = len(lines)
    for boundary_index, (start_line, heading) in enumerate(boundaries):
        end_line = boundaries[boundary_index + 1][0] if boundary_index + 1 < len(boundaries) else total_lines
        title, content = build_text_chapter(lines, start_line, end_line, heading)
        if not content.strip():
            continue
        chapters.append(
            {
                "title": title,
                "content": content,
                "path": f"line:{start_line + 1}",
            }
        )

    if chapters:
        return chapters

    return [{"title": infer_title_from_text(lines), "content": normalized, "path": ""}]


def detect_chapter_heading(line: str, line_index: int, matcher: str | None = None) -> str | None:
    """Return a normalized chapter title if the line looks like a chapter heading."""
    stripped = line.strip()
    if not stripped:
        return None

    markdown_heading = re.match(r"^(#{1,6})\s+(.*\S)\s*$", stripped)
    if markdown_heading:
        title = markdown_heading.group(2).strip()
        if looks_like_chapter_title(title, matcher=matcher):
            return title
        return None

    if line_index == 0 and len(stripped) > 120:
        return None

    if looks_like_chapter_title(stripped, matcher=matcher):
        return stripped

    return None


def looks_like_chapter_title(text: str, matcher: str | None = None) -> bool:
    """Heuristic for novel chapter headings in plain text or Markdown."""
    candidate = text.strip().strip("#").strip()
    if not candidate:
        return False
    if len(candidate) > 120:
        return False
    if matcher is not None:
        return bool(re.search(matcher, candidate))
    return bool(CHAPTER_MARKER_RE.match(candidate))


def build_text_chapter(lines: list[str], start_line: int, end_line: int, heading: str) -> tuple[str, str]:
    """Build a single chapter body from a heading-delimited line range."""
    content_lines = lines[start_line:end_line]
    if content_lines:
        first = content_lines[0].strip()
        if first.startswith("#"):
            content_lines = content_lines[1:]
        elif first == heading.strip():
            content_lines = content_lines[1:]

    content = "\n".join(content_lines).strip()
    return heading.strip(), content


def infer_title_from_text(lines: list[str]) -> str:
    """Fallback title for text sources with no explicit chapter headings."""
    for line in lines:
        stripped = line.strip()
        if stripped:
            return stripped[:80]
    return "Chapter 1"


def save_chapter(chapter: Chapter, output_dir: Path, source_file: str, digits: int) -> Path:
    """Write one chapter file with YAML frontmatter."""
    output_path = output_dir / f"ch{chapter.index:0{digits}d}.txt"
    frontmatter = {
        "index": chapter.index,
        "title": chapter.title,
        "length": len(chapter.content),
        "source_file": source_file,
        "path": chapter.path,
    }

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("---\n")
        yaml.dump(frontmatter, handle, allow_unicode=True, sort_keys=False)
        handle.write("---\n")
        handle.write(chapter.content)

    return output_path


def extract_text_from_html(html: str, remove_furigana: bool = True) -> str:
    """Extract plain text from HTML, optionally stripping ruby annotations."""
    from bs4 import BeautifulSoup, NavigableString

    soup = BeautifulSoup(html, "html.parser")

    for element in soup.find_all(["script", "style"]):
        element.decompose()

    for br in soup.find_all("br"):
        br.replace_with("\n")

    for paragraph in soup.find_all("p"):
        paragraph.append("\n\n")

    if remove_furigana:
        for ruby in soup.find_all("ruby"):
            base_text = "".join(rb.get_text() for rb in ruby.find_all("rb"))
            for child in ruby.children:
                if isinstance(child, NavigableString):
                    base_text += str(child)
            ruby.replace_with(base_text)

    lines = soup.get_text().split("\n")
    normalized_lines = [" ".join(line.split()) for line in lines]
    text = "\n".join(line for line in normalized_lines if line)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_title_from_html(html: str) -> str | None:
    """Extract a chapter title from HTML content."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")

    h1 = soup.find("h1")
    if h1:
        return extract_text_from_html(str(h1)).strip()

    title = soup.find("title")
    if title:
        return title.get_text(strip=True)

    strong = soup.find("strong")
    if strong:
        text = strong.get_text(strip=True)
        if text and len(text) < 100:
            return text

    return None
