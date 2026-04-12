from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from yorishiro.project import Project
from yorishiro.tasks.registry import ModelRegistry


def load_project(project_yaml: str) -> Project:
    root = Path(tempfile.mkdtemp())
    (root / "project.yaml").write_text(project_yaml, encoding="utf-8")
    return Project.load(root)


class ModelRegistryAgentTests(unittest.TestCase):
    def test_for_step_agent_uses_resolved_model_config(self) -> None:
        project = load_project(
            """project:
  name: Demo
  code: demo
sources: []
providers:
  openai_main:
    type: openai
    api_key_env: OPENAI_API_KEY
steps:
  novel.aliases:
    backend: pydantic-ai
    provider: openai_main
    model: gpt-5-mini
    thinking: low
    output_mode: native
"""
        )
        registry = ModelRegistry(project)

        with patch(
            "yorishiro.tasks.registry.build_agent_from_config", return_value="agent"
        ) as build_agent:
            agent = registry.for_step("novel.aliases").agent(
                output_type=dict,
                system_prompt="prompt",
            )

        self.assertEqual(agent, "agent")
        config = build_agent.call_args.args[0]
        self.assertEqual(config.backend, "pydantic-ai")
        self.assertEqual(config.provider, "openai")
        self.assertEqual(config.model, "gpt-5-mini")
        self.assertEqual(config.thinking, "low")
        self.assertEqual(config.output_mode, "native")

    def test_agent_runtime_rejects_instance_access(self) -> None:
        project = load_project(
            """project:
  name: Demo
  code: demo
sources: []
providers:
  openai_main:
    type: openai
steps:
  novel.aliases:
    backend: pydantic-ai
    provider: openai_main
    model: gpt-5-mini
"""
        )
        registry = ModelRegistry(project)

        with self.assertRaisesRegex(TypeError, "novel.aliases"):
            registry.for_step("novel.aliases").instance()


class ModelRegistryInstanceTests(unittest.TestCase):
    def test_for_step_instance_caches_single_step_runtime(self) -> None:
        project = load_project(
            """project:
  name: Demo
  code: demo
sources: []
steps:
  film.shots:
    backend: adaptive
"""
        )
        registry = ModelRegistry(project)

        sentinel = object()
        with patch.object(
            registry, "_build_shot_detector", return_value=sentinel
        ) as build_detector:
            runtime = registry.for_step("film.shots")
            first = runtime.instance()
            second = runtime.instance()

        self.assertIs(first, sentinel)
        self.assertIs(second, sentinel)
        self.assertEqual(build_detector.call_count, 1)

    def test_audio_separator_reads_inline_runtime_config(self) -> None:
        project = load_project(
            """project:
  name: Demo
  code: demo
sources: []
steps:
  film.audio.separate:
    backend: demucs
    model: htdemucs_ft
    device: cuda
    sample_rate: 48000
    processing_chunk_seconds: 90
    processing_overlap_seconds: 1.5
"""
        )
        registry = ModelRegistry(project)

        separator = registry.for_step("film.audio.separate").instance()

        self.assertEqual(separator.config.backend, "demucs")
        self.assertEqual(separator.config.model, "htdemucs_ft")
        self.assertEqual(separator.config.device, "cuda")
        self.assertEqual(separator.config.sample_rate, 48000)
        self.assertEqual(separator.config.processing_chunk_seconds, 90)
        self.assertEqual(separator.config.processing_overlap_seconds, 1.5)

    def test_music_analyzer_reads_nested_runtime_config(self) -> None:
        project = load_project(
            """project:
  name: Demo
  code: demo
sources: []
steps:
  film.audio.music:
    detect_lyrics: false
    separation:
      backend: demucs
      model: mdx_extra
    analysis:
      backend: essentia
"""
        )
        registry = ModelRegistry(project)

        analyzer = registry.for_step("film.audio.music").instance()

        self.assertEqual(analyzer.config.separation_backend, "demucs")
        self.assertEqual(analyzer.config.separation_model, "mdx_extra")
        self.assertEqual(analyzer.config.analysis_backend, "essentia")
        self.assertFalse(analyzer.config.detect_lyrics)

    def test_audio_steps_resolve_distinct_stage_runtimes(self) -> None:
        project = load_project(
            """project:
  name: Demo
  code: demo
sources: []
steps:
  film.audio.stt:
    backend: faster-whisper
    model: large-v3
    cpu_threads: 4
    num_workers: 2
  film.audio.speakers:
    speaker_similarity_threshold: 0.8
  film.audio.emotion:
    backend: emotion2vec
    model: emotion2vec/emotion2vec_plus_base
"""
        )
        registry = ModelRegistry(project)

        vad_runner = object()
        transcriber = object()
        speaker_attributor = object()
        emotion_analyzer = object()
        with (
            patch.object(
                registry, "_build_vad_runner", return_value=vad_runner
            ) as build_vad,
            patch.object(
                registry, "_build_transcriber", return_value=transcriber
            ) as build_transcriber,
            patch.object(
                registry, "_build_speaker_attributor", return_value=speaker_attributor
            ) as build_speakers,
            patch.object(
                registry, "_build_emotion_analyzer", return_value=emotion_analyzer
            ) as build_emotion,
        ):
            self.assertIs(registry.for_step("film.audio.vad").instance(), vad_runner)
            self.assertIs(registry.for_step("film.audio.stt").instance(), transcriber)
            self.assertIs(
                registry.for_step("film.audio.speakers").instance(), speaker_attributor
            )
            self.assertIs(
                registry.for_step("film.audio.emotion").instance(), emotion_analyzer
            )

        self.assertEqual(build_vad.call_count, 1)
        self.assertEqual(build_transcriber.call_count, 1)
        self.assertEqual(build_speakers.call_count, 1)
        self.assertEqual(build_emotion.call_count, 1)

    def test_transcriber_step_reads_grouping_and_filter_config(self) -> None:
        project = load_project(
            """project:
  name: Demo
  code: demo
sources: []
steps:
  film.audio.stt:
    backend: faster-whisper
    model: large-v3
    cpu_threads: 4
    num_workers: 2
    checkpoint_shard_size: 321
    group_max_duration_seconds: 18.5
    group_max_gap_seconds: 0.4
    min_confidence: -0.7
    max_chars_per_second: 19.0
"""
        )
        registry = ModelRegistry(project)

        runtime = registry.for_step("film.audio.stt")
        transcriber = runtime.instance()

        self.assertEqual(transcriber.config.stt_checkpoint_shard_size, 321)
        self.assertEqual(transcriber.config.stt_group_max_duration_seconds, 18.5)
        self.assertEqual(transcriber.config.stt_group_max_gap_seconds, 0.4)
        self.assertEqual(transcriber.config.stt_min_confidence, -0.7)
        self.assertEqual(transcriber.config.stt_max_chars_per_second, 19.0)

    def test_vad_step_reads_profile_config(self) -> None:
        project = load_project(
            """project:
  name: Demo
  code: demo
sources: []
steps:
  film.audio.vad:
    backend: silero-vad
    profile: precise
"""
        )
        registry = ModelRegistry(project)

        vad_runner = registry.for_step("film.audio.vad").instance()

        self.assertEqual(vad_runner.config.vad_profile, "precise")

    def test_instance_runtime_rejects_agent_access(self) -> None:
        project = load_project(
            """project:
  name: Demo
  code: demo
sources: []
steps:
  film.shots:
    backend: adaptive
"""
        )
        registry = ModelRegistry(project)

        with self.assertRaisesRegex(TypeError, "film.shots"):
            registry.for_step("film.shots").agent(
                output_type=dict, system_prompt="prompt"
            )
