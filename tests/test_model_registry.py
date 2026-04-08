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
steps:
  novel.aliases:
    model: alias-model
    output_mode: native
models:
  alias-model:
    provider: openai
    name: gpt-5-mini
    thinking: low
"""
        )
        registry = ModelRegistry(project)

        with patch("yorishiro.tasks.registry.build_agent_from_config", return_value="agent") as build_agent:
            agent = registry.for_step("novel.aliases").agent(
                output_type=dict,
                system_prompt="prompt",
            )

        self.assertEqual(agent, "agent")
        config = build_agent.call_args.args[0]
        self.assertEqual(config.provider, "openai")
        self.assertEqual(config.name, "gpt-5-mini")
        self.assertEqual(config.thinking, "low")
        self.assertEqual(config.output_mode, "native")

    def test_agent_runtime_rejects_instance_access(self) -> None:
        project = load_project(
            """project:
  name: Demo
  code: demo
sources: []
steps:
  novel.aliases:
    model: alias-model
models:
  alias-model:
    provider: openai
    name: gpt-5-mini
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
        with patch.object(registry, "_build_shot_detector", return_value=sentinel) as build_detector:
            runtime = registry.for_step("film.shots")
            first = runtime.instance()
            second = runtime.instance()

        self.assertIs(first, sentinel)
        self.assertIs(second, sentinel)
        self.assertEqual(build_detector.call_count, 1)

    def test_shared_speech_pipeline_cache_is_reused_across_steps(self) -> None:
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
  film.audio.diarize:
    diarization_model: diarizer
  film.audio.emotion:
    model: emotion2vec/emotion2vec_plus_base
models:
  diarizer:
    backend: pyannote
    model: pyannote/speaker-diarization-3.1
    batch_size: 32
    hf_token_env: HF_TOKEN
"""
        )
        registry = ModelRegistry(project)

        sentinel = object()
        with patch.object(registry, "_build_speech_pipeline", return_value=sentinel) as build_pipeline:
            vad_pipeline = registry.for_step("film.audio.vad").instance()
            stt_pipeline = registry.for_step("film.audio.stt").instance()
            emotion_pipeline = registry.for_step("film.audio.emotion").instance()

        self.assertIs(vad_pipeline, sentinel)
        self.assertIs(stt_pipeline, sentinel)
        self.assertIs(emotion_pipeline, sentinel)
        self.assertEqual(build_pipeline.call_count, 1)

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
            registry.for_step("film.shots").agent(output_type=dict, system_prompt="prompt")
