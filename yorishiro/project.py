"""Project configuration and path management for Yorishiro.

Usage:
    from yorishiro.project import Project
    
    project = Project.load(Path("projects/CPK"))
    source_dir = project.source_dir("cpk-novel")
    model_config = project.model_config("scene")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

ThinkingEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]

import yaml


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
    """Model configuration for a processing step."""
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
    base_url=None,  # None means use provider default
    api_key_env="YORISHIRO_API_KEY",
)


@dataclass
class Project:
    """Project configuration loaded from project.yaml."""
    root: Path
    name: str
    code: str
    sources: list[Source]
    model_default: ModelConfig
    model_steps: dict[str, ModelConfig]
    runtime: dict[str, Any]
    
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
        model_data = raw.get("model", {})
        runtime_data = raw.get("runtime", {})
        
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
        
        default_model = model_data.get("default", {})
        model_default = ModelConfig(
            provider=default_model.get("provider"),
            name=default_model.get("name"),
            thinking=default_model.get("thinking"),
            output_mode=default_model.get("output_mode"),
            base_url=default_model.get("base_url"),
            api_key_env=default_model.get("api_key_env"),
        )
        
        steps_data = model_data.get("steps", {})
        model_steps = {}
        for step_name, step_config in steps_data.items():
            model_steps[step_name] = ModelConfig(
                provider=step_config.get("provider"),
                name=step_config.get("name"),
                thinking=step_config.get("thinking"),
                output_mode=step_config.get("output_mode"),
                base_url=step_config.get("base_url"),
                api_key_env=step_config.get("api_key_env"),
            )
        
        runtime = {
            "batch_tokens": runtime_data.get("batch_tokens", 32000),
        }
        
        return cls(
            root=root,
            name=project_data.get("name", ""),
            code=project_data.get("code", ""),
            sources=sources,
            model_default=model_default,
            model_steps=model_steps,
            runtime=runtime,
        )
    
    def source_dir(self, source_id: str) -> Path:
        """Get the processed directory for a source."""
        return self.root / "processed" / source_id
    
    def souls_dir(self) -> Path:
        """Get the final souls output directory."""
        return self.root / "souls"
    
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
    
    def resolved_model_config(self, step: str | None = None) -> ModelConfig:
        """Get model configuration for a processing step.
        
        Precedence: step > default > fallback.
        
        Returns a ModelConfig with all fields resolved (thinking, output_mode, 
        base_url, api_key_env have fallback values; provider and name may still 
        be None if not configured anywhere).
        """
        # Start with fallback
        result = FALLBACK_CONFIG
        # Apply default config (default > fallback)
        result = result.merge(self.model_default)
        # Apply step config if specified (step > default > fallback)
        if step and step in self.model_steps:
            result = result.merge(self.model_steps[step])
        return result
    
    # Backwards compatibility alias
    def model_config(self, step: str) -> ModelConfig:
        """Deprecated: Use resolved_model_config instead."""
        return self.resolved_model_config(step)
    
    def list_characters(self, source_id: str) -> list[str]:
        """List all characters from a source's character_aliases.json."""
        aliases_path = self.source_dir(source_id) / "characters" / "character_aliases.json"
        if not aliases_path.exists():
            return []
        
        import json
        data = json.loads(aliases_path.read_text(encoding="utf-8"))
        return [k for k in data.keys() if k != "UNRESOLVED"]
    
    def list_chapters(self, source_id: str) -> list[int]:
        """List chapter indices for a source."""
        chapters_dir = self.source_dir(source_id) / "chapters"
        if not chapters_dir.exists():
            return []
        
        indices = []
        for f in chapters_dir.glob("ch*.txt"):
            try:
                idx = int(f.stem[2:])
                indices.append(idx)
            except ValueError:
                pass
        return sorted(indices)


def find_project(start: Path) -> Project | None:
    """Find project by walking up from start directory."""
    current = start.resolve()
    while current != current.parent:
        if (current / "project.yaml").exists():
            return Project.load(current)
        current = current.parent
    return None