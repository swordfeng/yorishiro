from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from yorishiro.synthesize import (
    collect_character_source_paths,
    discover_target_characters,
    load_all_notes_from_dirs,
    load_combined_insights,
)


class SynthesizeDiscoveryTests(unittest.TestCase):
    def test_discover_target_characters_uses_aliases_and_character_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            characters_dir = root / "characters"
            aliases_dir = root / "aliases"
            (characters_dir / "Alice").mkdir(parents=True)
            (characters_dir / "Bob").mkdir(parents=True)
            (aliases_dir / "Carol").mkdir(parents=True)

            (characters_dir / "Alice" / "scene001.json").write_text(
                json.dumps([{"chapter_index": 1, "scene_index": 0, "character": "Alice"}]),
                encoding="utf-8",
            )
            (characters_dir / "Bob" / "insights.md").write_text("Bob insight", encoding="utf-8")
            (aliases_dir / "Carol" / "insights.md").write_text("Carol alias insight", encoding="utf-8")

            found = discover_target_characters([characters_dir], [aliases_dir])

            self.assertEqual(found, ["Alice", "Bob", "Carol"])


class SynthesizeInputMergeTests(unittest.TestCase):
    def test_load_combined_insights_merges_character_and_alias_sections(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            characters_dir = root / "characters"
            aliases_dir = root / "aliases"
            (characters_dir / "Alice").mkdir(parents=True)
            (aliases_dir / "Alice").mkdir(parents=True)
            (characters_dir / "Alice" / "insights.md").write_text("From characters", encoding="utf-8")
            (aliases_dir / "Alice" / "insights.md").write_text("From aliases", encoding="utf-8")

            merged = load_combined_insights("Alice", [characters_dir], [aliases_dir])

            self.assertIn("## Character Insights", merged)
            self.assertIn("## Alias Insights", merged)
            self.assertIn("From characters", merged)
            self.assertIn("From aliases", merged)

    def test_notes_and_source_paths_tolerate_missing_and_invalid_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            characters_a = root / "a" / "characters"
            characters_b = root / "b" / "characters"
            aliases_dir = root / "aliases"
            (characters_a / "Alice").mkdir(parents=True)
            (characters_b / "Alice").mkdir(parents=True)
            (aliases_dir / "Alice").mkdir(parents=True)

            note = {"chapter_index": 1, "scene_index": 0, "character": "Alice"}
            (characters_a / "Alice" / "ch001.json").write_text(
                json.dumps([note]),
                encoding="utf-8",
            )
            (characters_b / "Alice" / "scene001.json").write_text(
                json.dumps([note, {"chapter_index": 1, "scene_index": 1, "character": "Alice"}]),
                encoding="utf-8",
            )
            (characters_b / "Alice" / "bad.json").write_text("{not-json", encoding="utf-8")
            (characters_b / "Alice" / "not_a_list.json").write_text(
                json.dumps({"chapter_index": 999}),
                encoding="utf-8",
            )
            (characters_a / "Alice" / "insights.md").write_text("From characters", encoding="utf-8")
            (aliases_dir / "Alice" / "insights.md").write_text("From aliases", encoding="utf-8")

            notes = load_all_notes_from_dirs([characters_a, characters_b], "Alice")
            source_paths = collect_character_source_paths("Alice", [characters_a, characters_b], [aliases_dir])

            self.assertEqual(len(notes), 2)
            self.assertEqual(
                [(n["chapter_index"], n["scene_index"]) for n in notes],
                [(1, 0), (1, 1)],
            )
            self.assertIn(characters_a / "Alice" / "insights.md", source_paths)
            self.assertIn(aliases_dir / "Alice" / "insights.md", source_paths)
            self.assertIn(characters_b / "Alice" / "scene001.json", source_paths)

