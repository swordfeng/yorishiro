from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from yorishiro.novel.alias_resolution import (
    AliasAssignment,
    BatchUpdateResult,
    CharacterEntry,
    GlobalState,
    MergeInstruction,
    MissedAliasResolution,
    SceneRecord,
    SeedFromSoulDocsResult,
    build_batches,
    find_missed,
    format_retry_prompt,
    load_all_scenes,
    process_all_batches,
    seed_from_insight_drafts,
    write_character_aliases,
    write_insight_drafts,
    apply_result,
)
from yorishiro.project import Project
from yorishiro.tasks.novel.aliases import NovelAliasesStep, NovelAliasesTask
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


def make_character(
    canonical_name: str,
    *,
    aliases: list[str] | None = None,
    facts: list[str] | None = None,
    possible_merge_candidates: list[str] | None = None,
    merge_notes: str = "",
    current_state: str = "",
    extra_notes: str = "",
) -> CharacterEntry:
    return CharacterEntry(
        canonical_name=canonical_name,
        aliases=aliases or [canonical_name],
        possible_merge_candidates=possible_merge_candidates or [],
        merge_notes=merge_notes,
        known_facts=facts or [],
        current_state=current_state,
        extra_notes=extra_notes,
    )


class LoadAllScenesTests(unittest.TestCase):
    def test_load_all_scenes_sorts_by_chapter_and_scene_and_skips_missing_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            ch2 = base / "ch002"
            ch2.mkdir()
            (ch2 / "scene_000.txt").write_text("Later chapter first scene", encoding="utf-8")
            (ch2 / "scenes_manifest.json").write_text(
                json.dumps(
                    {
                        "scenes": [
                            {"scene_index": 1, "characters": ["B"], "location": "Room", "time": "Night"},
                            {"scene_index": 0, "characters": ["A"], "location": "Hall", "time": "Day"},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            ch1 = base / "ch001"
            ch1.mkdir()
            (ch1 / "scene_000.txt").write_text("Earlier chapter scene", encoding="utf-8")
            (ch1 / "scenes_manifest.json").write_text(
                json.dumps({"scenes": [{"scene_index": 0, "characters": ["C"], "location": "Park", "time": "Noon"}]}),
                encoding="utf-8",
            )

            scenes = load_all_scenes(base)

            self.assertEqual([(scene.chapter, scene.scene_index) for scene in scenes], [("ch001", 0), ("ch002", 0)])
            self.assertEqual(scenes[1].text, "Later chapter first scene")


class BuildBatchesTests(unittest.TestCase):
    def test_build_batches_keeps_large_scene_intact(self) -> None:
        scenes = [
            SceneRecord(chapter="ch000", scene_index=0, characters=["A"], location="", time="", text="a" * 600),
            SceneRecord(chapter="ch000", scene_index=1, characters=["B"], location="", time="", text="b" * 600),
        ]

        batches = build_batches(scenes, batch_tokens=10)

        self.assertEqual(len(batches), 2)
        self.assertEqual([scene.scene_index for scene in batches[0]], [0])
        self.assertEqual([scene.scene_index for scene in batches[1]], [1])


class ApplyResultTests(unittest.TestCase):
    def test_apply_result_merges_occurrences_and_clears_unresolved(self) -> None:
        state = GlobalState(
            characters={
                "Alice": make_character("Alice", possible_merge_candidates=["Alicia"]),
                "Alicia": make_character("Alicia"),
            },
            occurrence_map={("ch001", 0, "Alicia"): "Alicia"},
            unresolved_map={("ch001", 0, "Ally"): "unclear"},
        )
        result = BatchUpdateResult(
            updated_characters=[make_character("Alice", aliases=["Alice", "Ally"])],
            alias_assignments=[AliasAssignment(chapter="ch001", scene_index=0, alias="Ally", canonical_name="Alice")],
            unresolved_aliases=[],
            merges=[MergeInstruction(keep="Alice", absorb="Alicia", reason="same person")],
            knowledge_summary="Summary",
            batch_notes="",
        )

        apply_result(result, state)

        self.assertIn("Alice", state.characters)
        self.assertNotIn("Alicia", state.characters)
        self.assertEqual(state.occurrence_map[("ch001", 0, "Alicia")], "Alice")
        self.assertEqual(state.occurrence_map[("ch001", 0, "Ally")], "Alice")
        self.assertNotIn(("ch001", 0, "Ally"), state.unresolved_map)
        self.assertNotIn("Alicia", state.characters["Alice"].possible_merge_candidates)
        self.assertEqual(state.knowledge_summary, "Summary")


class MissedAliasTests(unittest.TestCase):
    def test_find_missed_only_returns_aliases_not_assigned_or_unresolved(self) -> None:
        batch = [
            SceneRecord(chapter="ch001", scene_index=0, characters=["A", "B", "C"], location="", time="", text="scene"),
        ]
        state = GlobalState(
            occurrence_map={("ch001", 0, "A"): "Alice"},
            unresolved_map={("ch001", 0, "B"): "unclear"},
        )
        result = BatchUpdateResult(
            updated_characters=[],
            alias_assignments=[],
            unresolved_aliases=[],
            merges=[],
            knowledge_summary="",
            batch_notes="",
        )

        missed = find_missed(result, batch, state)

        self.assertEqual(missed, [("ch001", 0, "C")])

    def test_format_retry_prompt_includes_scene_context_once(self) -> None:
        scene = SceneRecord(chapter="ch001", scene_index=2, characters=["A", "B"], location="Cafe", time="Morning", text="Context text")
        prompt = format_retry_prompt(
            [("ch001", 2, "A"), ("ch001", 2, "B")],
            {("ch001", 2): scene},
            GlobalState(),
        )

        self.assertEqual(prompt.count("Scene context"), 1)
        self.assertIn("「A」", prompt)
        self.assertIn("「B」", prompt)
        self.assertIn("Context text", prompt)


class ProcessAllBatchesTests(unittest.IsolatedAsyncioTestCase):
    async def test_process_all_batches_retries_missed_aliases(self) -> None:
        batch = [[SceneRecord(chapter="ch001", scene_index=0, characters=["Ally"], location="Room", time="Night", text="Ally spoke.")]]
        batch_agent = FakeAgent(
            [
                BatchUpdateResult(
                    updated_characters=[make_character("Alice", aliases=["Alice", "Ally"])],
                    alias_assignments=[],
                    unresolved_aliases=[],
                    merges=[],
                    knowledge_summary="Story so far",
                    batch_notes="",
                )
            ]
        )
        retry_agent = FakeAgent(
            [
                MissedAliasResolution(
                    alias_assignments=[AliasAssignment(chapter="ch001", scene_index=0, alias="Ally", canonical_name="Alice")],
                    unresolved_aliases=[],
                )
            ]
        )

        state = await process_all_batches(batch, GlobalState(), batch_agent, retry_agent)

        self.assertEqual(state.occurrence_map[("ch001", 0, "Ally")], "Alice")
        self.assertEqual(state.knowledge_summary, "Story so far")
        self.assertEqual(len(retry_agent.prompts), 1)
        self.assertIn("Missed Aliases", retry_agent.prompts[0])


class SeedAndWriteTests(unittest.IsolatedAsyncioTestCase):
    async def test_seed_from_insight_drafts_parses_existing_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            char_dir = base / "Alice"
            char_dir.mkdir()
            (char_dir / "insights.md").write_text("# Alice\n\nFact", encoding="utf-8")
            agent = FakeAgent([SeedFromSoulDocsResult(characters=[make_character("Alice", facts=["Fact"])])])

            state = await seed_from_insight_drafts(base, agent)

            self.assertEqual(list(state.characters.keys()), ["Alice"])
            self.assertEqual(state.characters["Alice"].known_facts, ["Fact"])

    def test_write_character_aliases_and_insight_drafts_emit_expected_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            state = GlobalState(
                characters={
                    "Alice": make_character(
                        "Alice",
                        aliases=["Alice", "Ally"],
                        facts=["Brave"],
                        current_state="Waiting",
                        possible_merge_candidates=["Alicia"],
                        merge_notes="Might be same person",
                        extra_notes="Important lead",
                    )
                },
                occurrence_map={
                    ("ch002", 1, "Ally"): "Alice",
                    ("ch001", 0, "Alice"): "Alice",
                },
                unresolved_map={("ch003", 0, "Girl"): "too vague"},
            )

            aliases_path = base / "character_aliases.json"
            write_character_aliases(state, aliases_path)
            data = json.loads(aliases_path.read_text(encoding="utf-8"))
            self.assertEqual(data["Alice"][0], {"chapter": "ch001", "scene": 0, "alias": "Alice"})
            self.assertEqual(data["Alice"][1], {"chapter": "ch002", "scene": 1, "alias": "Ally"})
            self.assertEqual(data["UNRESOLVED"][0]["reason"], "too vague")

            write_insight_drafts(state, base / "characters")
            draft = (base / "characters" / "Alice" / "insights.md").read_text(encoding="utf-8")
            self.assertIn("## Aliases", draft)
            self.assertIn("## Known Facts", draft)
            self.assertIn("## Current State", draft)
            self.assertIn("## Possible Identity Overlap", draft)
            self.assertIn("## Notes", draft)


class NovelAliasesStepTests(unittest.TestCase):
    def test_step_reads_batch_tokens_from_project_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            scenes_dir = root / "processed" / "novel-src" / "steps" / "scenes" / "ch000"
            scenes_dir.mkdir(parents=True)
            (scenes_dir / "scenes_manifest.json").write_text(json.dumps({"scenes": []}), encoding="utf-8")
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
  novel.aliases:
    backend: pydantic-ai
    provider: openai_main
    model: gpt-5-mini
    batch_tokens: 12345
""",
                encoding="utf-8",
            )

            project = Project.load(root)
            task = NovelAliasesStep(project, "novel-src", registry=ModelRegistry(project)).tasks()[0]

            self.assertIsInstance(task, NovelAliasesTask)
            assert isinstance(task, NovelAliasesTask)
            self.assertEqual(task._batch_tokens, 12345)
            self.assertEqual(task.input_paths()[0], project.config_path)


if __name__ == "__main__":
    unittest.main()
