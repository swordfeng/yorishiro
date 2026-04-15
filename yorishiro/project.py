"""Project configuration and path management for Yorishiro.

Usage:
    from yorishiro.project import Project

    project = Project.load(Path("projects/CPK"))
    step_dir = project.step_dir("cpk-novel", "scenes")
    model_config = project.resolved_model_config("novel.scenes")
"""

from __future__ import annotations

import os
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

    backend: str | None = None
    provider: str | None = None
    model: str | None = None
    thinking: ThinkingEffort | None = None
    output_mode: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    api_key_env: str | None = None

    def merge(self, other: ModelConfig) -> ModelConfig:
        """Merge with another config. Values in `other` take precedence over `self`."""
        return ModelConfig(
            backend=other.backend if other.backend is not None else self.backend,
            provider=other.provider if other.provider is not None else self.provider,
            model=other.model if other.model is not None else self.model,
            thinking=other.thinking if other.thinking is not None else self.thinking,
            output_mode=(
                other.output_mode if other.output_mode is not None else self.output_mode
            ),
            base_url=other.base_url if other.base_url is not None else self.base_url,
            api_key=other.api_key if other.api_key is not None else self.api_key,
            api_key_env=(
                other.api_key_env if other.api_key_env is not None else self.api_key_env
            ),
        )


# Fallback defaults for optional fields
FALLBACK_CONFIG = ModelConfig(
    thinking="medium",
    output_mode="tool",
    base_url=None,
    api_key=None,
    api_key_env="YORISHIRO_API_KEY",
)

# Fields that belong to ModelConfig (cloud models only)
_CLOUD_MODEL_FIELDS = {
    "backend",
    "provider",
    "model",
    "thinking",
    "output_mode",
    "base_url",
    "api_key",
    "api_key_env",
}


@dataclass
class Project:
    """Project configuration loaded from project.yaml."""

    config_path: Path
    root: Path
    name: str
    code: str
    sources: list[Source]
    providers: dict[str, dict[str, Any]]
    steps: dict[str, dict[str, Any]]
    step_groups: dict[str, list[str]]
    hf_token: str | None = None
    hf_token_env: str = "YORISHIRO_HF_TOKEN"

    @classmethod
    def load(cls, path: Path) -> Project:
        """Load project from a config file or directory."""
        if path.is_file():
            config_path = path
            root = path.parent
        else:
            root = path
            config_path = root / "project.yaml"

        if not config_path.exists():
            raise FileNotFoundError(f"project.yaml not found at {config_path}")

        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

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

        providers = raw.get("providers", {})
        steps = raw.get("steps", {})
        step_groups = raw.get("step_groups", {})

        hf_cfg = project_data.get("hf_token") or raw.get("hf_token")
        hf_token_val: str | None = None
        hf_token_env_val: str = "YORISHIRO_HF_TOKEN"
        if isinstance(hf_cfg, dict):
            hf_token_val = hf_cfg.get("value")
            hf_token_env_val = hf_cfg.get("env", "YORISHIRO_HF_TOKEN")
        elif isinstance(hf_cfg, str):
            hf_token_val = hf_cfg

        project = cls(
            config_path=config_path,
            root=root,
            name=project_data.get("name", ""),
            code=project_data.get("code", ""),
            sources=sources,
            providers=providers,
            steps=steps,
            step_groups=step_groups,
            hf_token=hf_token_val,
            hf_token_env=hf_token_env_val,
        )
        project.login_huggingface()
        return project

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def resolved_hf_token(self) -> str | None:
        """Return the HuggingFace token, checking direct value then env var."""
        if self.hf_token:
            return self.hf_token
        return os.environ.get(self.hf_token_env)

    def login_huggingface(self) -> None:
        """Authenticate with HuggingFace using the project-level token.

        After this call, all HF libraries (transformers, pyannote, wespeaker)
        can download gated models without explicit token arguments.
        """
        token = self.resolved_hf_token()
        if not token:
            return
        from huggingface_hub import login

        login(token=token, add_to_git_credential=False)

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
        """Return the raw config mapping for a step."""
        return dict(self.steps.get(step_id, {}))

    def resolved_model_config(self, step_id: str) -> ModelConfig:
        """Return resolved ModelConfig for a cloud LLM step.

        Merges: fallback defaults → provider profile → step-local overrides.
        The step must declare `backend: pydantic-ai`, a provider profile reference,
        and a concrete `model` id.
        """
        step_cfg = self.step_config(step_id)
        backend = step_cfg.get("backend")
        if backend != "pydantic-ai":
            raise ValueError(
                f"Step '{step_id}' must set backend: pydantic-ai "
                "to use cloud agent runtime"
            )

        provider_profile = step_cfg.get("provider")
        if not provider_profile:
            raise ValueError(
                f"Step '{step_id}' with backend 'pydantic-ai' is missing provider"
            )

        provider_cfg = self.providers.get(str(provider_profile))
        if provider_cfg is None:
            raise ValueError(
                f"Step '{step_id}' references unknown provider profile "
                f"'{provider_profile}'"
            )

        provider_type = provider_cfg.get("type")
        if not provider_type:
            raise ValueError(
                f"Provider profile '{provider_profile}' for step '{step_id}' "
                "is missing required field 'type'"
            )

        if not step_cfg.get("model"):
            raise ValueError(
                f"Step '{step_id}' with backend 'pydantic-ai' is missing model"
            )

        merged_cfg = {**provider_cfg}
        merged_cfg["provider"] = provider_type
        merged_cfg.pop("type", None)
        for key, value in step_cfg.items():
            if key != "provider":
                merged_cfg[key] = value

        cloud_fields = {
            k: merged_cfg[k] for k in _CLOUD_MODEL_FIELDS if k in merged_cfg
        }
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
            aliases_path = (
                self.source_dir(source_id) / "characters" / "character_aliases.json"
            )
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
            key=lambda p: (
                int(match.group()) if (match := re.search(r"\d+", p.stem)) else 0
            ),
        )
