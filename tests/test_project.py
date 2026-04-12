from __future__ import annotations

import argparse
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from yorishiro.agent_utils import (
    _build_model_settings,
    add_model_args,
    build_agent,
    build_agent_from_config,
)
from yorishiro.project import ModelConfig, Project


class ProjectLoadTests(unittest.TestCase):
    def test_load_project_yaml_from_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "project.yaml").write_text(
                "project:\n  name: Primary\n  code: primary\nsources: []\n",
                encoding="utf-8",
            )

            project = Project.load(root)

            self.assertEqual(project.name, "Primary")
            self.assertEqual(project.config_path, root / "project.yaml")

    def test_load_raises_when_project_yaml_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            with self.assertRaisesRegex(FileNotFoundError, "project.yaml not found"):
                Project.load(root)

    def test_load_explicit_file_path_records_config_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_path = root / "project.yaml"
            config_path.write_text(
                "project:\n  name: Example\n  code: example\nsources: []\n",
                encoding="utf-8",
            )

            project = Project.load(config_path)

            self.assertEqual(project.config_path, config_path)
            self.assertEqual(project.root, root)


class ProjectConfigTests(unittest.TestCase):
    def _load_project(self, project_yaml: str) -> Project:
        root = Path(tempfile.mkdtemp())
        (root / "project.yaml").write_text(project_yaml, encoding="utf-8")
        return Project.load(root)

    def test_resolved_model_config_merges_provider_profile_and_step_overrides(self) -> None:
        project = self._load_project(
            """project:
  name: Demo
  code: demo
sources: []
providers:
  openrouter_main:
    type: openrouter
    base_url: https://openrouter.example/api/v1
    api_key_env: OPENROUTER_API_KEY
steps:
  novel.scenes:
    backend: pydantic-ai
    provider: openrouter_main
    model: anthropic/claude-sonnet-4-6
    thinking: high
    output_mode: native
    api_key: inline-secret
"""
        )

        config = project.resolved_model_config("novel.scenes")

        self.assertEqual(config.backend, "pydantic-ai")
        self.assertEqual(config.provider, "openrouter")
        self.assertEqual(config.model, "anthropic/claude-sonnet-4-6")
        self.assertEqual(config.thinking, "high")
        self.assertEqual(config.output_mode, "native")
        self.assertEqual(config.base_url, "https://openrouter.example/api/v1")
        self.assertEqual(config.api_key, "inline-secret")
        self.assertEqual(config.api_key_env, "OPENROUTER_API_KEY")

    def test_resolved_model_config_rejects_unknown_provider_profile(self) -> None:
        project = self._load_project(
            """project:
  name: Demo
  code: demo
sources: []
steps:
  novel.scenes:
    backend: pydantic-ai
    provider: missing_profile
    model: gpt-5-mini
"""
        )

        with self.assertRaisesRegex(
            ValueError, "unknown provider profile 'missing_profile'"
        ):
            project.resolved_model_config("novel.scenes")

    def test_resolved_model_config_rejects_provider_profile_missing_type(self) -> None:
        project = self._load_project(
            """project:
  name: Demo
  code: demo
sources: []
providers:
  openai_main:
    api_key_env: OPENAI_API_KEY
steps:
  novel.scenes:
    backend: pydantic-ai
    provider: openai_main
    model: gpt-5-mini
"""
        )

        with self.assertRaisesRegex(ValueError, "missing required field 'type'"):
            project.resolved_model_config("novel.scenes")

    def test_resolved_model_config_rejects_missing_model(self) -> None:
        project = self._load_project(
            """project:
  name: Demo
  code: demo
sources: []
providers:
  openai_main:
    type: openai
steps:
  novel.scenes:
    backend: pydantic-ai
    provider: openai_main
"""
        )

        with self.assertRaisesRegex(ValueError, "missing model"):
            project.resolved_model_config("novel.scenes")


class AgentConfigTests(unittest.TestCase):
    def test_build_agent_from_config_uses_inline_api_key(self) -> None:
        with patch("yorishiro.agent_utils.build_agent", return_value="agent") as build_agent:
            agent = build_agent_from_config(
                ModelConfig(
                    backend="pydantic-ai",
                    provider="openai",
                    model="gpt-5-mini",
                    api_key="inline-secret",
                ),
                output_type=dict,
                system_prompt="prompt",
            )

        self.assertEqual(agent, "agent")
        self.assertEqual(build_agent.call_args.kwargs["api_key"], "inline-secret")
        self.assertEqual(build_agent.call_args.kwargs["provider_name"], "openai")
        self.assertEqual(build_agent.call_args.kwargs["model_name"], "gpt-5-mini")

    def test_build_agent_from_config_reads_api_key_env_when_inline_key_absent(self) -> None:
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "env-secret"}, clear=True),
            patch("yorishiro.agent_utils.build_agent", return_value="agent") as build_agent,
        ):
            agent = build_agent_from_config(
                ModelConfig(
                    backend="pydantic-ai",
                    provider="openai",
                    model="gpt-5-mini",
                    api_key_env="OPENAI_API_KEY",
                ),
                output_type=dict,
                system_prompt="prompt",
            )

        self.assertEqual(agent, "agent")
        self.assertEqual(build_agent.call_args.kwargs["api_key"], "env-secret")

    def test_openai_compatible_providers_map_thinking_to_reasoning_effort(self) -> None:
        for provider in ["openai", "openrouter", "deepseek", "moonshotai", "ollama"]:
            with self.subTest(provider=provider):
                self.assertEqual(
                    _build_model_settings(provider, "medium", "tool"),
                    {"openai_reasoning_effort": "medium"},
                )

    def test_openai_reasoning_effort_normalizes_extremes(self) -> None:
        self.assertEqual(
            _build_model_settings("openai", "minimal", "tool"),
            {"openai_reasoning_effort": "low"},
        )
        self.assertEqual(
            _build_model_settings("openai", "xhigh", "tool"),
            {"openai_reasoning_effort": "high"},
        )

    def test_anthropic_thinking_uses_budget_tokens(self) -> None:
        self.assertEqual(
            _build_model_settings("anthropic", "medium", "prompted"),
            {"anthropic_thinking": {"type": "enabled", "budget_tokens": 4096}},
        )

    def test_anthropic_thinking_rejects_tool_output_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "Anthropic thinking is incompatible"):
            _build_model_settings("anthropic", "medium", "tool")

    def test_unknown_provider_ignores_thinking(self) -> None:
        self.assertIsNone(_build_model_settings("google", "medium", "tool"))

    def test_thinking_none_disables_model_settings(self) -> None:
        self.assertIsNone(_build_model_settings("openai", "none", "tool"))
        self.assertIsNone(_build_model_settings("anthropic", "none", "prompted"))

    def test_build_agent_passes_model_settings_to_agent(self) -> None:
        fake_provider = object()
        with (
            patch("pydantic_ai.providers.infer_provider_class", return_value=lambda **kwargs: fake_provider),
            patch("pydantic_ai.models.infer_model", return_value="resolved-model"),
            patch("yorishiro.agent_utils.Agent", return_value="agent") as agent_cls,
        ):
            agent = build_agent(
                model_name="gpt-5-mini",
                provider_name="openrouter",
                api_key="secret",
                base_url="",
                output_type=dict,
                system_prompt="prompt",
                thinking="medium",
                output_mode="tool",
            )

        self.assertEqual(agent, "agent")
        self.assertEqual(
            agent_cls.call_args.kwargs["model_settings"],
            {"openai_reasoning_effort": "medium"},
        )

    def test_add_model_args_accepts_extended_thinking_levels(self) -> None:
        parser = argparse.ArgumentParser()
        add_model_args(parser)
        args = parser.parse_args(["--thinking", "minimal"])
        self.assertEqual(args.thinking, "minimal")
