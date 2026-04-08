from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from yorishiro.novel.scene_segmentation import (
    SceneData,
    SceneSegment,
    SceneSegmentationConfig,
    SegmentationResult,
    append_scene,
    find_end_offset,
    parse_chapter_file,
    segment_chapter,
    verify_coverage,
)
from yorishiro.project import Project
from yorishiro.tasks.novel.scenes import NovelScenesStep, NovelScenesTask
from yorishiro.tasks.registry import ModelRegistry


class FakeAgent:
    def __init__(self, outputs: list[SegmentationResult]) -> None:
        self._outputs = list(outputs)
        self.prompts: list[str] = []

    async def run(self, prompt: str) -> SimpleNamespace:
        self.prompts.append(prompt)
        if not self._outputs:
            raise AssertionError("FakeAgent was called more times than expected")
        return SimpleNamespace(output=self._outputs.pop(0))


def build_scene(
    *,
    location: str = "School",
    time: str = "Afternoon",
    characters: list[str] | None = None,
    boundary_type: str = "narrative_break",
    end_text: str = "",
) -> SceneSegment:
    return SceneSegment(
        location=location,
        time=time,
        characters=characters or ["A"],
        boundary_type=boundary_type,
        end_text=end_text,
    )


class ParseChapterFileTests(unittest.TestCase):
    def test_parse_chapter_file_extracts_frontmatter_and_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "chapter.txt"
            path.write_text("---\nindex: 7\ntitle: Reunion\n---\nBody text.\n", encoding="utf-8")

            metadata, content = parse_chapter_file(path)

            self.assertEqual(metadata["index"], 7)
            self.assertEqual(metadata["title"], "Reunion")
            self.assertEqual(content, "Body text.\n")

    def test_parse_chapter_file_falls_back_to_plain_text_without_frontmatter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "chapter.txt"
            path.write_text("Plain body only.", encoding="utf-8")

            metadata, content = parse_chapter_file(path)

            self.assertEqual(metadata, {})
            self.assertEqual(content, "Plain body only.")


class SceneSegmentationConfigTests(unittest.TestCase):
    def test_from_step_config_accepts_string_numbers(self) -> None:
        config = SceneSegmentationConfig.from_step_config(
            {"initial_chunk_size": "1024", "max_chunk_size": "4096"}
        )

        self.assertEqual(config.initial_chunk_size, 1024)
        self.assertEqual(config.max_chunk_size, 4096)

    def test_from_step_config_rejects_non_positive_initial_chunk_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "initial_chunk_size must be > 0"):
            SceneSegmentationConfig.from_step_config({"initial_chunk_size": 0})

    def test_from_step_config_rejects_max_smaller_than_initial(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_chunk_size must be >="):
            SceneSegmentationConfig.from_step_config(
                {"initial_chunk_size": 2000, "max_chunk_size": 1000}
            )


class FindEndOffsetTests(unittest.TestCase):
    def test_find_end_offset_ignores_whitespace_differences(self) -> None:
        text = "Alpha  \n  Beta   Gamma"

        end = find_end_offset(text, "Alpha Beta", 0)

        self.assertEqual(text[:end], "Alpha  \n  Beta")

    def test_find_end_offset_handles_literal_backslash_n_in_model_output(self) -> None:
        text = "Alpha\nBeta"

        end = find_end_offset(text, "Alpha\\nBeta", 0)

        self.assertEqual(end, len(text))

    def test_find_end_offset_supports_fuzzy_match_for_long_near_match(self) -> None:
        text = "The lantern flickered softly in the quiet hallway."
        typo = "The lantern flickared softly in the quiet hallway."

        end = find_end_offset(text, typo, 0)

        self.assertEqual(end, len(text))

    def test_find_end_offset_raises_for_missing_short_end_text(self) -> None:
        with self.assertRaisesRegex(ValueError, "Could not locate end_text"):
            find_end_offset("abcdefghij", "xyz", 0)

    def test_find_end_offset_respects_search_from_for_repeated_text(self) -> None:
        text = "END marker. filler. END marker."
        first_end = find_end_offset(text, "END marker.", 0)
        second_end = find_end_offset(text, "END marker.", first_end)

        self.assertEqual(text[:first_end], "END marker.")
        self.assertEqual(text[first_end:second_end], " filler. END marker.")


class AppendSceneTests(unittest.TestCase):
    def test_append_scene_assigns_incrementing_indexes(self) -> None:
        scenes: list[SceneData] = []

        first = append_scene(scenes, 0, 5, build_scene(end_text="dummy"))
        second = append_scene(scenes, 5, 9, build_scene(end_text="dummy"))

        self.assertEqual(first.scene_index, 0)
        self.assertEqual(second.scene_index, 1)

    def test_append_scene_rejects_gaps(self) -> None:
        scenes = [SceneData(scene_index=0, start_offset=0, end_offset=5, location="A", time="T")]

        with self.assertRaisesRegex(ValueError, "Continuity gap"):
            append_scene(scenes, 6, 10, build_scene(end_text="dummy"))

    def test_append_scene_rejects_empty_ranges(self) -> None:
        with self.assertRaisesRegex(ValueError, "Empty/inverted scene"):
            append_scene([], 4, 4, build_scene(end_text="dummy"))


class SegmentChapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_segment_chapter_carries_incomplete_scene_to_next_batch(self) -> None:
        chapter_text = "Alpha scene ends here.\n\nBeta scene closes the chapter."
        agent = FakeAgent(
            [
                SegmentationResult(
                    scenes=[
                        build_scene(
                            location="Hall",
                            time="Noon",
                            characters=["Aya"],
                            end_text="Alpha scene ends here.",
                        ),
                        build_scene(
                            location="Street",
                            time="Evening",
                            characters=["Ben"],
                            end_text="",
                        ),
                    ],
                    has_more=True,
                    summary="",
                ),
                SegmentationResult(
                    scenes=[
                        build_scene(
                            location="Street",
                            time="Evening",
                            characters=["Ben"],
                            end_text="",
                        )
                    ],
                    has_more=False,
                    summary="Earlier summary",
                ),
            ]
        )

        scenes = await segment_chapter(chapter_text, agent, SceneSegmentationConfig(initial_chunk_size=24, max_chunk_size=48))

        self.assertEqual(len(scenes), 2)
        self.assertEqual(chapter_text[scenes[0].start_offset:scenes[0].end_offset], "Alpha scene ends here.")
        self.assertEqual(chapter_text[scenes[1].start_offset:scenes[1].end_offset], "\n\nBeta scene closes the chapter.")
        verify_coverage(scenes, len(chapter_text))
        self.assertEqual(len(agent.prompts), 2)
        self.assertIn("[Previously processed — last few scenes BEFORE cursor]", agent.prompts[1])
        self.assertIn("Alpha scene ends here.", agent.prompts[1])

    async def test_segment_chapter_retries_after_empty_scene_batch(self) -> None:
        chapter_text = "One complete scene."
        agent = FakeAgent(
            [
                SegmentationResult(scenes=[], has_more=True, summary=""),
                SegmentationResult(
                    scenes=[build_scene(location="Room", time="Night", end_text="")],
                    has_more=False,
                    summary="",
                ),
            ]
        )

        scenes = await segment_chapter(chapter_text, agent, SceneSegmentationConfig(initial_chunk_size=8, max_chunk_size=32))

        self.assertEqual(len(scenes), 1)
        self.assertEqual(scenes[0].start_offset, 0)
        self.assertEqual(scenes[0].end_offset, len(chapter_text))
        self.assertEqual(len(agent.prompts), 2)

    @unittest.expectedFailure
    async def test_segment_chapter_reports_actual_failed_end_text_on_retry(self) -> None:
        chapter_text = "First scene ends here. Second scene ends here."
        bad_first_end_text = "THIS TEXT IS NOT PRESENT"
        later_end_text = "Second scene ends here."
        agent = FakeAgent(
            [
                SegmentationResult(
                    scenes=[
                        build_scene(end_text=bad_first_end_text),
                        build_scene(end_text=later_end_text),
                    ],
                    has_more=False,
                    summary="",
                ),
                SegmentationResult(
                    scenes=[build_scene(location="Hall", time="Noon", end_text="")],
                    has_more=False,
                    summary="",
                ),
            ]
        )

        await segment_chapter(chapter_text, agent, SceneSegmentationConfig(initial_chunk_size=16, max_chunk_size=64))

        self.assertIn(bad_first_end_text, agent.prompts[1])


class NovelScenesStepTests(unittest.TestCase):
    def test_step_reads_chunk_size_overrides_from_project_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            chapter_dir = root / "processed" / "novel-src" / "steps" / "chapters"
            chapter_dir.mkdir(parents=True)
            (chapter_dir / "ch000.txt").write_text("---\nindex: 0\ntitle: One\n---\nAlpha", encoding="utf-8")
            (root / "project.yaml").write_text(
                """project:
  name: Demo
  code: demo
sources:
  - id: novel-src
    type: novel
    path: raw/novel.md
steps:
  novel.scenes:
    model: scene-model
    initial_chunk_size: 1234
    max_chunk_size: 5678
models:
  scene-model:
    provider: openai
    name: gpt-5-mini
""",
                encoding="utf-8",
            )

            project = Project.load(root)
            task = NovelScenesStep(project, "novel-src", registry=ModelRegistry(project)).tasks()[0]

            self.assertIsInstance(task, NovelScenesTask)
            assert isinstance(task, NovelScenesTask)
            self.assertEqual(
                task._segmentation_config,
                SceneSegmentationConfig(initial_chunk_size=1234, max_chunk_size=5678),
            )


if __name__ == "__main__":
    unittest.main()
