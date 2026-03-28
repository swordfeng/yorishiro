"""Shared utilities for pydantic-ai agent construction and CLI argument setup.

All Yorishiro extraction pipeline scripts share the same provider/model/key wiring.
Import add_model_args, resolve_api_key, and build_agent from here instead of duplicating them.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import TypeVar

from pydantic_ai import Agent

_OutputT = TypeVar("_OutputT")


def add_model_args(parser: argparse.ArgumentParser) -> None:
    """Add standard model/provider CLI arguments to an ArgumentParser."""
    parser.add_argument(
        "--provider",
        default=os.environ.get("YORISHIRO_PROVIDER", "openrouter"),
        help="Provider name (env: YORISHIRO_PROVIDER, default: openrouter)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("YORISHIRO_MODEL", "anthropic/claude-opus-4-6"),
        help="Model name (env: YORISHIRO_MODEL, default: anthropic/claude-opus-4-6)",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("YORISHIRO_BASE_URL", ""),
        help="API base URL override (env: YORISHIRO_BASE_URL, empty = provider default)",
    )
    parser.add_argument(
        "--api-key-env",
        default="YORISHIRO_API_KEY",
        help="Name of the env var holding the API key (default: YORISHIRO_API_KEY)",
    )
    parser.add_argument(
        "--thinking",
        default=os.environ.get("YORISHIRO_THINKING", "medium"),
        choices=["none", "low", "medium", "high"],
        help="Model thinking/reasoning effort (env: YORISHIRO_THINKING, default: medium)",
    )
    parser.add_argument(
        "--output-mode",
        default=os.environ.get("YORISHIRO_OUTPUT_MODE", "tool"),
        choices=["tool", "native", "prompted"],
        help="Structured output mode: tool (default), native, prompted (env: YORISHIRO_OUTPUT_MODE)",
    )


def resolve_api_key(args: argparse.Namespace) -> str:
    """Read API key from the env var named by args.api_key_env. Exits on missing."""
    env_var = getattr(args, "api_key_env", "YORISHIRO_API_KEY")
    key = os.environ.get(env_var)
    if not key:
        print(f"Error: {env_var} environment variable is not set", file=sys.stderr)
        sys.exit(1)
    return key


def build_agent(
    model_name: str,
    provider_name: str,
    api_key: str,
    base_url: str,
    output_type: type[_OutputT],
    system_prompt: str,
    thinking: str = "medium",
    output_mode: str = "tool",
    tools: list | None = None,
) -> Agent[None, _OutputT]:
    """Build a pydantic-ai Agent with configurable output type, system prompt, and optional tools.

    output_type: a Pydantic model class (will be wrapped per output_mode).
    tools: list of plain Python functions to register as agent tools.
    """
    from pydantic_ai.models import infer_model
    from pydantic_ai.providers import infer_provider_class
    from pydantic_ai.capabilities import Thinking
    from pydantic_ai.output import NativeOutput, PromptedOutput, ToolOutput

    cls = infer_provider_class(provider_name)
    kwargs: dict = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    provider = cls(**kwargs)

    model = infer_model(f"{provider_name}:{model_name}", provider_factory=lambda _: provider)

    capabilities = []
    if thinking != "none":
        capabilities.append(Thinking(effort=thinking))  # type: ignore[arg-type]

    if output_mode == "native":
        wrapped_output = NativeOutput(output_type)
    elif output_mode == "prompted":
        wrapped_output = PromptedOutput(output_type)
    else:
        wrapped_output = ToolOutput(output_type)

    agent_kwargs: dict = {
        "model": model,
        "output_type": wrapped_output,
        "system_prompt": system_prompt,
        "capabilities": capabilities,
    }
    if tools:
        agent_kwargs["tools"] = tools

    return Agent(**agent_kwargs)
