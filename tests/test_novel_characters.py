from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from yorishiro.novel.character_extraction import (
    SoulDocAppend,
    build_alias_canonical_lookup,
    build_batches,
    load_aliases,
    load_all_scenes,
    make_extraction_models,
    append_insights,
    process_all_batches,
    write_chapter_notes,
)
from yorishiro.project import Project
from yorishiro.tasks.novel.characters import NovelCharactersStep, NovelCharactersTask
from yorishiro.tasks.registry import ModelRegistry


class FakeAgent:
    def __init__(self, outputs: list[object]) -> None:
        self._outputs = list(outputs)
        self.prompts: list[str] = []

    async def run(self, prompt: str) -> SimpleNamespace:
        self.prompts.append(prompt)
        if not self._outputs:
            raise AssertionError("FakeAgent was called more times than expected")
        return SimpleNamespace(output=self._outputs.pop(0))


class AliasLoadingTests(unittest.TestCase):
    def test_load_aliases_excludes_unresolved_and_builds_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "character_aliases.json"
            path.write_text(
                json.dumps(
                    {
                        "Alice": [{"chapter": "ch001", "scene": 0, "alias": "Ally"}],
                        "UNRESOLVED": [{"chapter": "ch001", "scene": 1, "alias": "Girl", "reason": "unknown"}],
                    }
                ),
                encoding="utf-8",
            )

            aliases = load_aliases(path)
            lookup = build_alias_canonical_lookup(aliases)

            self.assertEqual(list(aliases.keys()), ["Alice"])
            self.assertEqual(lookup[("ch001", 0, "Ally")], "Alice")


class SceneLoadingTests(unittest.TestCase):
    def test_load_all_scenes_applies_canonical_names_and_target_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            chapter_dir = base / "ch001"
            chapter_dir.mkdir()
            (chapter_dir / "scene_000.txt").write_text("Scene text", encoding="utf-8")
            (chapter_dir / "scenes_manifest.json").write_text(
                json.dumps(
                    {
                        "chapter_index": 1,
                        "scenes": [
                            {
                                "scene_index": 0,
                                "location": "Cafe",
                                "time": "Noon",
                                "characters": ["Ally", "Bob"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            aliases = {"Alice": [{"chapter": "ch001", "scene": 0, "alias": "Ally"}]}

            scenes = load_all_scenes(base, aliases, ["Alice"])

            self.assertEqual(len(scenes), 1)
            self.assertEqual([(c.alias, c.canonical) for c in scenes[0].characters], [("Ally", "Alice"), ("Bob", "Bob")])
            self.assertEqual([(c.alias, c.canonical) for c in scenes[0].target_characters], [("Ally", "Alice")])


class ModelAndBatchTests(unittest.TestCase):
    def test_make_extraction_models_constrains_character_field(self) -> None:
        CharacterSceneNoteModel, BatchExtractionResultModel = make_extraction_models(["Alice", "Bob"])

        enum_values = CharacterSceneNoteModel.model_json_schema()["properties"]["character"]["enum"]
        self.assertEqual(enum_values, ["Alice", "Bob"])
        self.assertIn("notes", BatchExtractionResultModel.model_json_schema()["properties"])

    def test_build_batches_keeps_large_scene_intact(self) -> None:
        aliases = {"Alice": [{"chapter": "ch001", "scene": 0, "alias": "Ally"}]}
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            for idx, text in enumerate(["a" * 500, "b" * 500]):
                chapter_dir = base / f"ch00{idx + 1}"
                chapter_dir.mkdir()
                (chapter_dir / "scene_000.txt").write_text(text, encoding="utf-8")
                (chapter_dir / "scenes_manifest.json").write_text(
                    json.dumps({"chapter_index": idx, "scenes": [{"scene_index": 0, "characters": ["Ally"]}]}),
                    encoding="utf-8",
                )
            scenes = load_all_scenes(base, aliases, ["Alice"])

            batches = build_batches(scenes, batch_tokens=10)

            self.assertEqual(len(batches), 2)
            self.assertEqual([scene.chapter for scene in batches[0]], ["ch001"])
            self.assertEqual([scene.chapter for scene in batches[1]], ["ch002"])


class OutputWritingTests(unittest.TestCase):
    def test_write_chapter_notes_sorts_by_scene_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = write_chapter_notes(
                Path(tmp_dir),
                "Alice",
                "ch001",
                [
                    {"scene_index": 2, "note": "later"},
                    {"scene_index": 0, "note": "earlier"},
                ],
            )

            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual([entry["scene_index"] for entry in saved], [0, 2])

    def test_append_insights_writes_only_populated_sections(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            append_insights(
                Path(tmp_dir),
                SoulDocAppend(
                    canonical_name="Alice",
                    negative_constraints=["Never lies"],
                    behavioral_patterns=["Hums when nervous"],
                    arc_notes="Grows more direct",
                ),
                1,
                3,
            )

            text = (Path(tmp_dir) / "Alice" / "insights.md").read_text(encoding="utf-8")
            self.assertIn("## Batch 1/3", text)
            self.assertIn("### Negative Constraints", text)
            self.assertIn("### Behavioral Patterns", text)
            self.assertIn("### Arc Notes", text)
            self.assertNotIn("### Relationship Insights", text)


class ProcessAllBatchesTests(unittest.IsolatedAsyncioTestCase):
    async def test_process_all_batches_writes_notes_and_appends_insights(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            characters_dir = root / "characters"
            characters_dir.mkdir()
            (characters_dir / "Alice").mkdir()
            (characters_dir / "Alice" / "insights.md").write_text("Existing insight", encoding="utf-8")

            aliases = {"Alice": [{"chapter": "ch001", "scene": 0, "alias": "Ally"}]}
            scene_dir = root / "scenes" / "ch001"
            scene_dir.mkdir(parents=True)
            (scene_dir / "scene_000.txt").write_text("Ally said hello.", encoding="utf-8")
            (scene_dir / "scenes_manifest.json").write_text(
                json.dumps(
                    {
                        "chapter_index": 1,
                        "scenes": [{"scene_index": 0, "location": "Room", "time": "Night", "characters": ["Ally"]}],
                    }
                ),
                encoding="utf-8",
            )
            scenes = load_all_scenes(root / "scenes", aliases, ["Alice"])
            _, BatchExtractionResultModel = make_extraction_models(["Alice"])
            extraction = BatchExtractionResultModel.model_validate(
                {
                    "notes": [
                        {
                            "character": "Alice",
                            "chapter_index": 1,
                            "scene_index": 0,
                            "active_persona": "self",
                            "dialogue_samples": ["こんにちは"],
                            "language_traits": "丁寧",
                            "emotional_state": "落ち着いている",
                            "inferred_motivation": "様子を見る",
                            "internal_conflict": "",
                            "relationships": [],
                            "actions_taken": "挨拶した",
                            "actions_avoided": "深入りしなかった",
                            "decision_logic": "慎重だった",
                            "arc_marker": "導入",
                            "knowledge_scope": {"facts_revealed": ["相手がいる"], "facts_hidden": []},
                            "comfort_mechanisms": [],
                            "repeated_expressions": [],
                            "persona_shifts": [],
                            "sensory_triggers": [],
                        }
                    ],
                    "soul_doc_appends": [
                        {
                            "canonical_name": "Alice",
                            "behavioral_patterns": ["慎重に距離を測る (ch001/s00)"],
                            "arc_notes": "最初は慎重",
                        }
                    ],
                }
            )
            agent = FakeAgent([extraction])

            await process_all_batches([scenes], ["Alice"], characters_dir, characters_dir, agent)

            notes_path = characters_dir / "Alice" / "ch001.json"
            self.assertTrue(notes_path.exists())
            notes = json.loads(notes_path.read_text(encoding="utf-8"))
            self.assertEqual(notes[0]["character"], "Alice")

            insights = (characters_dir / "Alice" / "insights.md").read_text(encoding="utf-8")
            self.assertIn("### Behavioral Patterns", insights)
            self.assertIn("慎重に距離を測る", insights)
            self.assertEqual(len(agent.prompts), 1)
            self.assertIn("### Alice", agent.prompts[0])


class NovelCharactersStepTests(unittest.TestCase):
    def test_step_reads_batch_tokens_from_project_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "project.yaml").write_text(
                """project:
  name: Demo
  code: demo
sources:
  - id: novel-src
    type: novel
    path: raw/novel.md
providers:
  openai_main:
    type: openai
steps:
  novel.characters:
    backend: pydantic-ai
    provider: openai_main
    model: gpt-5-mini
    batch_tokens: 12345
""",
                encoding="utf-8",
            )

            project = Project.load(root)
            task = NovelCharactersStep(project, "novel-src", registry=ModelRegistry(project)).tasks()[0]

            self.assertIsInstance(task, NovelCharactersTask)
            assert isinstance(task, NovelCharactersTask)
            self.assertEqual(task._batch_tokens, 12345)
            self.assertEqual(
                task.input_paths(),
                [project.step_dir("novel-src", "aliases") / "character_aliases.json"],
            )


if __name__ == "__main__":
    unittest.main()
