"""Shared utilities for pydantic-ai agent construction and CLI argument setup.

All Yorishiro pipeline scripts share the same provider/model/key wiring.
Import add_model_args and build_agent_from_args from here instead of duplicating them.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, TypeVar, cast

import tiktoken
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

from yorishiro.project import ModelConfig, ThinkingEffort

_encoding = None

_OutputT = TypeVar("_OutputT")


def estimate_tokens(text: str) -> int:
    """Estimate token count using cl100k_base encoding (GPT-4/3.5 tokenizer)."""
    global _encoding
    if _encoding is None:
        _encoding = tiktoken.get_encoding("cl100k_base")
    return len(_encoding.encode(text))


def add_model_args(parser: argparse.ArgumentParser) -> None:
    """Add standard model/provider CLI arguments to an ArgumentParser.

    All arguments default to None. Values should come from project.yaml
    or be specified explicitly on the command line.
    """
    parser.add_argument(
        "--provider",
        default=None,
        help="Upstream provider name override (e.g., openrouter, openai)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name (e.g., anthropic/claude-opus-4-6)",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="API base URL override (optional)",
    )
    parser.add_argument(
        "--api-key-env",
        default=None,
        help="Name of the env var holding the API key",
    )
    parser.add_argument(
        "--thinking",
        default=None,
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        help="Model thinking/reasoning effort",
    )
    parser.add_argument(
        "--output-mode",
        default=None,
        choices=["tool", "native", "prompted"],
        help="Structured output mode",
    )


_OPENAI_REASONING_PROVIDERS = {
    "openai",
    "openrouter",
    "deepseek",
    "moonshotai",
    "ollama",
}


def _normalize_openai_reasoning_effort(thinking: ThinkingEffort) -> str | None:
    if thinking == "none":
        return None
    if thinking == "minimal":
        return "low"
    if thinking == "xhigh":
        return "high"
    return thinking


def _anthropic_thinking_budget(thinking: ThinkingEffort) -> int | None:
    return {
        "none": None,
        "minimal": 1024,
        "low": 2048,
        "medium": 4096,
        "high": 8192,
        "xhigh": 16384,
    }[thinking]


def _build_model_settings(
    provider_name: str,
    thinking: ThinkingEffort,
    output_mode: str,
) -> ModelSettings | None:
    provider_key = provider_name.lower()
    if provider_key in _OPENAI_REASONING_PROVIDERS:
        effort = _normalize_openai_reasoning_effort(thinking)
        if effort is None:
            return None
        return cast(ModelSettings, {"openai_reasoning_effort": effort})

    if provider_key == "anthropic":
        budget = _anthropic_thinking_budget(thinking)
        if budget is None:
            return None
        if output_mode == "tool":
            raise ValueError(
                "Anthropic thinking is incompatible with output_mode='tool'; "
                "use output_mode='prompted' or set thinking='none'"
            )
        settings: dict[str, Any] = {
            "anthropic_thinking": {
                "type": "enabled",
                "budget_tokens": budget,
            }
        }
        return cast(ModelSettings, settings)

    return None


def build_agent(
    model_name: str,
    provider_name: str,
    api_key: str,
    base_url: str,
    output_type: type[_OutputT],
    system_prompt: str,
    thinking: ThinkingEffort = "medium",
    output_mode: str = "tool",
    tools: list | None = None,
) -> Agent[None, _OutputT]:
    """Build a pydantic-ai Agent with configurable output type, system prompt, and optional tools.

    output_type: a Pydantic model class (will be wrapped per output_mode).
    tools: list of plain Python functions to register as agent tools.
    """
    from pydantic_ai.models import infer_model
    from pydantic_ai.providers import infer_provider_class
    from pydantic_ai.output import NativeOutput, PromptedOutput, ToolOutput

    cls = infer_provider_class(provider_name)
    kwargs: dict = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    provider = cls(**kwargs)

    model = infer_model(f"{provider_name}:{model_name}", provider_factory=lambda _: provider)
    model_settings = _build_model_settings(provider_name, thinking, output_mode)

    if output_mode == "native":
        wrapped_output = NativeOutput(output_type)
    elif output_mode == "prompted":
        wrapped_output = PromptedOutput(output_type)
    else:
        wrapped_output = ToolOutput(output_type)

    agent = Agent(
        model=model,
        output_type=wrapped_output,
        system_prompt=system_prompt,
        model_settings=model_settings,
        tools=tools or [],
    )
    # pydantic-ai's NativeOutput/PromptedOutput/ToolOutput constructors are typed via
    # TypeAliasType with type_params, which ty cannot unify back to OutputDataT, so the
    # agent is inferred as Agent[None, str]. The actual output type is _OutputT since
    # the wrappers are constructed with output_type: type[_OutputT].
    return cast(Agent[None, _OutputT], agent)


def build_agent_from_config(
    config: ModelConfig,
    output_type: type[_OutputT],
    system_prompt: str,
    tools: list | None = None,
) -> Agent[None, _OutputT]:
    """Build a pydantic-ai Agent from resolved ModelConfig only.

    This is the non-CLI path used by the task runtime layer. The provided config
    is expected to already include project-level fallbacks via
    Project.resolved_model_config().
    """
    backend = config.backend
    provider = config.provider
    model_name = config.model
    thinking = config.thinking or "medium"
    output_mode = config.output_mode or "tool"
    base_url = config.base_url or ""
    api_key_env = config.api_key_env or "YORISHIRO_API_KEY"

    if backend != "pydantic-ai":
        raise ValueError(f"Unsupported agent backend '{backend}' in ModelConfig")
    if provider is None:
        raise ValueError("No provider specified in ModelConfig")
    if model_name is None:
        raise ValueError("No model specified in ModelConfig")

    api_key = config.api_key
    if not api_key:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            print(
                f"Error: {api_key_env} environment variable is not set",
                file=sys.stderr,
            )
            sys.exit(1)

    return build_agent(
        model_name=model_name,
        provider_name=provider,
        api_key=api_key,
        base_url=base_url,
        output_type=output_type,
        system_prompt=system_prompt,
        thinking=thinking,
        output_mode=output_mode,
        tools=tools,
    )


def build_agent_from_args(
    args: argparse.Namespace,
    output_type: type[_OutputT],
    system_prompt: str,
    config: ModelConfig | None = None,
    tools: list | None = None,
) -> Agent[None, _OutputT]:
    """Build a pydantic-ai Agent from CLI arguments and optional config.

    Precedence: args > config > fallback.

    Config should come from Project.resolved_model_config() which includes
    fallbacks for optional fields.

    Args:
        args: Argument namespace from argparse (all fields default to None).
        output_type: A Pydantic model class for structured output.
        system_prompt: The system prompt for the agent.
        config: Optional ModelConfig from project.yaml (with fallbacks applied).
        tools: Optional list of plain Python functions to register as agent tools.

    Returns:
        Configured Agent instance.

    Raises:
        ValueError: If provider or model not specified.
    """
    # Resolve with precedence: args > config > fallback
    provider = args.provider or (config.provider if config else None)
    model_name = args.model or (config.model if config else None)
    thinking = args.thinking or (config.thinking if config else None)
    output_mode = args.output_mode or (config.output_mode if config else None)
    base_url = args.base_url or (config.base_url if config else None)
    api_key_env = args.api_key_env or (config.api_key_env if config else None)
    api_key = None if args.api_key_env else (config.api_key if config else None)

    # Apply final fallbacks for optional fields
    thinking = thinking or "medium"
    output_mode = output_mode or "tool"
    api_key_env = api_key_env or "YORISHIRO_API_KEY"

    # Validate required fields
    if provider is None:
        raise ValueError(
            "No provider specified. Use --provider CLI argument or "
            "set the step provider profile in project.yaml"
        )
    if model_name is None:
        raise ValueError(
            "No model specified. Use --model CLI argument or "
            "set the step model in project.yaml"
        )

    return build_agent_from_config(
        ModelConfig(
            backend="pydantic-ai",
            provider=provider,
            model=model_name,
            thinking=thinking,
            output_mode=output_mode,
            base_url=base_url,
            api_key=api_key,
            api_key_env=api_key_env,
        ),
        output_type=output_type,
        system_prompt=system_prompt,
        tools=tools,
    )
