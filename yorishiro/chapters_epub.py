"""Extract chapters from EPUB to YAML frontmatter files.

Usage:
    uv run python -m yorishiro.chapters_epub --project <project_dir> --source <source_id>
    uv run python -m yorishiro.chapters_epub --source <source_id>  # uses cwd as project

Output:
    {output_dir}/
    ├── ch000.txt
    ├── ch001.txt
    └── ...

File format (YAML frontmatter):
    ---
    index: 0
    title: "第一章 降临"
    length: 15234
    source_file: "CPK.epub"
    path: "OEBPS/chapter001.xhtml"
    ---
    [chapter text content...]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import yaml

from yorishiro.backup import ProjectBackup
from yorishiro.project import Project
from yorishiro.utils import is_output_stale


@dataclass
class Chapter:
    """Represents a chapter from an epub."""
    index: int
    title: str
    content: str
    path: str = ""


def extract_chapters(
    epub_path: Path,
    output_dir: Path | None = None,
    remove_furigana: bool = True,
) -> list[Path]:
    """Extract chapters from EPUB and save as YAML frontmatter files.
    
    This is the main function that can be called programmatically.
    
    Args:
        epub_path: Path to the EPUB file.
        output_dir: Directory to save chapter files. Defaults to epub_path.parent / "chapters".
        
    Returns:
        List of paths to saved chapter files.
    """
    if not epub_path.exists():
        raise FileNotFoundError(f"EPUB file not found: {epub_path}")
    
    if output_dir is None:
        output_dir = epub_path.parent / "chapters"
    
    output_dir.mkdir(parents=True, exist_ok=True)

    chapters_list = list(parse_epub(epub_path, remove_furigana=remove_furigana))
    digits = max(3, len(str(len(chapters_list))))
    saved_files = []

    for chapter in chapters_list:
        output_path = save_chapter(chapter, output_dir, epub_path.name, digits=digits)
        saved_files.append(output_path)

    return saved_files


def parse_epub(epub_path: Path, remove_furigana: bool = True) -> Iterator[Chapter]:
    """Parse an EPUB file and yield chapters."""
    import ebooklib
    from ebooklib import epub

    book = epub.read_epub(str(epub_path))
    chapter_count = 0

    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            html_content = item.get_content().decode('utf-8', errors='ignore')
            text = extract_text_from_html(html_content, remove_furigana=remove_furigana)
            
            if text.strip():
                title = extract_title_from_html(html_content) or f"Chapter {chapter_count + 1}"
                item_path = getattr(item, 'href', None) or getattr(item, 'file_name', str(item.get_name()))
                
                yield Chapter(
                    index=chapter_count,
                    title=title,
                    content=text,
                    path=item_path
                )
                chapter_count += 1


def save_chapter(
    chapter: Chapter,
    output_dir: Path,
    source_file: str,
    digits: int = 3,
) -> Path:
    """Save a chapter to a YAML frontmatter file."""
    filename = f"ch{chapter.index:0{digits}d}.txt"
    output_path = output_dir / filename
    
    # Build frontmatter
    frontmatter = {
        "index": chapter.index,
        "title": chapter.title,
        "length": len(chapter.content),
        "source_file": source_file,
        "path": chapter.path,
    }
    
    # Write YAML frontmatter + content
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("---\n")
        yaml.dump(frontmatter, f, allow_unicode=True, sort_keys=False)
        f.write("---\n")
        f.write(chapter.content)
    
    return output_path


def extract_text_from_html(html: str, remove_furigana: bool = True) -> str:
    """Extract plain text from HTML using BeautifulSoup.
    
    Args:
        html: HTML content
        remove_furigana: If True, remove ruby/furigana annotations (default: True)
    """
    from bs4 import BeautifulSoup, NavigableString
    
    soup = BeautifulSoup(html, 'html.parser')
    
    # Remove script and style elements
    for element in soup.find_all(['script', 'style']):
        element.decompose()
    
    # Handle line breaks
    # Convert <br> to newlines
    for br in soup.find_all('br'):
        br.replace_with('\n')
    
    # Convert </p><p> to double newlines (paragraph separation)
    for p in soup.find_all('p'):
        # Add newlines after each paragraph
        p.append('\n\n')
    
    # Remove furigana/ruby annotations if requested
    if remove_furigana:
        for ruby in soup.find_all('ruby'):
            # Replace ruby element with just the base text (rb), removing rt (furigana)
            base_text = ''
            for rb in ruby.find_all('rb'):
                base_text += rb.get_text()
            # Also get direct text content not in rb/rt
            for child in ruby.children:
                if isinstance(child, NavigableString):
                    base_text += str(child)
            ruby.replace_with(base_text)
    
    # Get text
    text = soup.get_text()
    
    # Normalize whitespace: collapse multiple spaces/tabs, but preserve newlines
    lines = text.split('\n')
    normalized_lines = []
    for line in lines:
        # Collapse multiple spaces within a line
        normalized = ' '.join(line.split())
        if normalized:
            normalized_lines.append(normalized)
    
    # Join with newlines and clean up excessive blank lines
    text = '\n'.join(normalized_lines)
    # Replace 3+ consecutive newlines with 2 newlines
    import re
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()


def extract_title_from_html(html: str) -> str | None:
    """Extract title from HTML content (h1, title, or first strong)."""
    from bs4 import BeautifulSoup
    
    soup = BeautifulSoup(html, 'html.parser')
    
    # Try h1 first
    h1 = soup.find('h1')
    if h1:
        return extract_text_from_html(str(h1)).strip()
    
    # Try title tag
    title = soup.find('title')
    if title:
        return title.get_text(strip=True)
    
    # Try strong tag
    strong = soup.find('strong')
    if strong:
        text = strong.get_text(strip=True)
        if text and len(text) < 100:
            return text
    
    return None


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Extract chapters from EPUB to YAML frontmatter files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  uv run python -m yorishiro.chapters_epub --project projects/CPK --source cpk-novel\n"
            "  uv run python -m yorishiro.chapters_epub --source cpk-novel  # uses cwd as project\n"
        ),
    )
    
    parser.add_argument(
        "--project",
        type=Path,
        default=None,
        help="Project directory (default: current directory)",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Source ID within project",
    )
    parser.add_argument(
        "--keep-furigana",
        action="store_true",
        default=False,
        help="Keep ruby/furigana annotations (default: remove them)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing chapter files",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip backup snapshot after processing",
    )
    args = parser.parse_args()

    project_path = args.project if args.project else Path.cwd()
    project = Project.load(project_path)
    
    source = project.get_source(args.source)
    if not source:
        print(f"Error: Source '{args.source}' not found in project", file=sys.stderr)
        sys.exit(1)
    
    epub_path = project.get_source_path(args.source)
    output_dir = project.source_dir(args.source) / "chapters"
    remove_furigana = not args.keep_furigana
    
    if not args.force and output_dir.exists():
        chapter_files = list(output_dir.glob("ch*.txt"))
        if chapter_files and not is_output_stale(output_dir / ".chapters_done", [epub_path]):
            print(f"Skipping: {output_dir} already has {len(chapter_files)} chapters (use --force to re-extract)")
            sys.exit(0)

    try:
        saved_files = extract_chapters(epub_path, output_dir, remove_furigana=remove_furigana)

        (output_dir / ".chapters_done").touch()

        if not args.no_backup:
            backup = ProjectBackup(project.root)
            backup.snapshot(f"chapters-{args.source}")

        print(f"Extracted {len(saved_files)} chapters from {epub_path.name}")
        print(f"Output: {output_dir}")
        for path in saved_files[:5]:
            print(f"  - {path.name}")
        if len(saved_files) > 5:
            print(f"  ... and {len(saved_files) - 5} more")
            
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
