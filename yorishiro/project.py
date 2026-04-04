"""Project configuration and path management for Yorishiro.

Usage:
    from yorishiro.project import Project

    project = Project.load(Path("projects/CPK"))
    step_dir = project.step_dir("cpk-novel", "scenes")
    model_config = project.resolved_model_config("novel.scenes")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

ThinkingEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]

AUTHORITY_LEVELS = ("PRIMARY", "SECONDARY", "RUMOR")


@dataclass
class Source:
    """A source material in the project."""
    id: str
    type: str
    path: str
    authority: str = "PRIMARY"
    config: dict = field(default_factory=dict)


@dataclass
class ModelConfig:
    """Model configuration for a cloud LLM processing step."""
    provider: str | None = None
    name: str | None = None
    thinking: ThinkingEffort | None = None
    output_mode: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None

    def merge(self, other: ModelConfig) -> ModelConfig:
        """Merge with another config. Values in `other` take precedence over `self`."""
        return ModelConfig(
            provider=other.provider or self.provider,
            name=other.name or self.name,
            thinking=other.thinking or self.thinking,
            output_mode=other.output_mode or self.output_mode,
            base_url=other.base_url or self.base_url,
            api_key_env=other.api_key_env or self.api_key_env,
        )


# Fallback defaults for optional fields
FALLBACK_CONFIG = ModelConfig(
    thinking="medium",
    output_mode="tool",
    base_url=None,
    api_key_env="YORISHIRO_API_KEY",
)

# Fields that belong to ModelConfig (cloud models only)
_CLOUD_MODEL_FIELDS = {"provider", "name", "thinking", "output_mode", "base_url", "api_key_env"}


@dataclass
class Project:
    """Project configuration loaded from project.yaml."""
    root: Path
    name: str
    code: str
    sources: list[Source]
    # New schema: flat model registry + per-step config
    models: dict[str, dict]         # models.<name> → raw definition dict
    steps: dict[str, dict]          # steps.<step_id> → raw config dict
    step_groups: dict[str, list[str]]

    @classmethod
    def load(cls, path: Path) -> Project:
        """Load project from directory containing project.yaml."""
        if path.is_file():
            config_path = path
            root = path.parent
        else:
            config_path = path / "project.yaml"
            root = path

        if not config_path.exists():
            raise FileNotFoundError(f"project.yaml not found at {config_path}")

        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))

        project_data = raw.get("project", {})
        sources_data = raw.get("sources", [])

        sources = [
            Source(
                id=s.get("id"),
                type=s.get("type", "unknown"),
                path=s.get("path", ""),
                authority=s.get("authority", "PRIMARY"),
                config=s.get("config", {}),
            )
            for s in sources_data
        ]

        models = raw.get("models", {})
        steps = raw.get("steps", {})
        step_groups = raw.get("step_groups", {})

        return cls(
            root=root,
            name=project_data.get("name", ""),
            code=project_data.get("code", ""),
            sources=sources,
            models=models,
            steps=steps,
            step_groups=step_groups,
        )

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def step_dir(self, source_id: str, step_name: str) -> Path:
        """Get the output directory for a specific step of a source.

        e.g. step_dir("cpk-novel", "scenes") → processed/cpk-novel/steps/scenes/
        """
        return self.root / "processed" / source_id / "steps" / step_name

    def cross_dir(self, step_name: str = "") -> Path:
        """Get the cross-source directory, optionally with a sub-step name."""
        base = self.root / "cross"
        return base / step_name if step_name else base

    def source_dir(self, source_id: str) -> Path:
        """Get the processed root directory for a source."""
        return self.root / "processed" / source_id

    def souls_dir(self) -> Path:
        """Get the final souls output directory."""
        return self.root / "souls"

    # ------------------------------------------------------------------
    # Source helpers
    # ------------------------------------------------------------------

    def get_source(self, source_id: str) -> Source | None:
        """Get source configuration by ID."""
        for source in self.sources:
            if source.id == source_id:
                return source
        return None

    def get_source_path(self, source_id: str) -> Path:
        """Get the raw source file path (may be external)."""
        source = self.get_source(source_id)
        if not source:
            raise ValueError(f"Source '{source_id}' not found")

        path = Path(source.path)
        if path.is_absolute():
            return path
        return self.root / path

    # ------------------------------------------------------------------
    # Model config resolution
    # ------------------------------------------------------------------

    def step_config(self, step_id: str) -> dict[str, Any]:
        """Return merged config for a step: step overrides → model definition.

        The 'model' key is resolved and its fields merged underneath step fields.
        """
        step = dict(self.steps.get(step_id, {}))
        model_name = step.pop("model", None)
        model_def = dict(self.models.get(model_name, {})) if model_name else {}
        # step fields override model definition fields
        return {**model_def, **step}

    def resolved_model_config(self, step_id: str) -> ModelConfig:
        """Return resolved ModelConfig for a cloud LLM step.

        Merges: step config → model definition → FALLBACK_CONFIG.
        Only cloud-model fields are extracted (provider, name, thinking, …).
        """
        cfg = self.step_config(step_id)
        cloud_fields = {k: cfg[k] for k in _CLOUD_MODEL_FIELDS if k in cfg}
        step_mc = ModelConfig(**cloud_fields)
        return FALLBACK_CONFIG.merge(step_mc)

    def local_model_config(self, step_id: str) -> dict[str, Any]:
        """Return merged config dict for a local ML step."""
        return self.step_config(step_id)

    # ------------------------------------------------------------------
    # Content helpers
    # ------------------------------------------------------------------

    def list_characters(self, source_id: str) -> list[str]:
        """List all characters from a source's character_aliases.json."""
        aliases_path = self.step_dir(source_id, "aliases") / "character_aliases.json"
        if not aliases_path.exists():
            # fall back to legacy path
            aliases_path = self.source_dir(source_id) / "characters" / "character_aliases.json"
        if not aliases_path.exists():
            return []

        import json
        data = json.loads(aliases_path.read_text(encoding="utf-8"))
        return [k for k in data.keys() if k != "UNRESOLVED"]

    def list_chapters(self, source_id: str) -> list[Path]:
        """Return sorted list of chapter file paths."""
        chapters_dir = self.step_dir(source_id, "chapters")
        if not chapters_dir.exists():
            # fall back to legacy path
            chapters_dir = self.source_dir(source_id) / "chapters"
        if not chapters_dir.exists():
            return []
        chapters = list(chapters_dir.glob("ch*.txt"))
        return sorted(
            chapters,
            key=lambda p: int(match.group()) if (match := re.search(r"\d+", p.stem)) else 0,
        )
