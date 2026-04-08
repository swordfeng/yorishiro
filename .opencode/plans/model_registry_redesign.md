# ModelRegistry Redesign Plan

## Summary

Redesign `ModelRegistry` around a step-scoped runtime API:

- `registry.for_step(step_id) -> StepRuntime`
- `StepRuntime.agent(output_type, system_prompt)`
- `StepRuntime.instance()`

The goal is to hide local-vs-remote from steps, scale across different model interaction styles, and stop adding one public accessor per step.

## Design Goals

- Keep step code focused on its interaction style, not model deployment details.
- Remove step-facing use of `cloud_config()`, `local_config()`, and bespoke `get_*()` registry methods.
- Support growth by registering step runtime specs declaratively instead of extending the public registry interface per step.
- Preserve current config semantics and runtime behavior as much as possible.

## Public API

### `ModelRegistry.for_step(step_id: str) -> StepRuntime`

Returns a runtime facade for a specific step.

Behavior:

- validates that the step is registered
- carries the step id for error messages
- resolves runtime behavior from an internal per-step spec table

### `StepRuntime.agent(output_type, system_prompt, tools=None)`

For agent-backed steps.

Behavior:

- resolves the step’s `ModelConfig` internally
- builds an agent directly from resolved config
- does not require the step to create fake CLI args
- raises `TypeError` if called for an instance-backed step

### `StepRuntime.instance()`

For local model/service-backed steps.

Behavior:

- lazily builds the concrete local runtime dependency
- caches the built instance according to the step spec
- raises `TypeError` if called for an agent-backed step

## Internal Structure

### 1. Keep config resolution inside the registry

`ModelRegistry` should continue to use `Project.step_config()` and `Project.resolved_model_config()` internally.

Guideline:

- step code reads `Project.step_config(step_id)` only for non-model step parameters such as batch sizes or chunk sizes
- step code does not ask whether a model is local or remote

### 2. Add a declarative step spec table

Use a private spec table keyed by step id.

Each spec should define:

- runtime kind: `agent` or `instance`
- factory function for instance-backed steps
- cache key strategy for instance-backed steps

This makes adding a new step require:

1. registering its runtime spec
2. using `for_step(step_id)` at the call site

It does not require:

1. adding a new public `get_*()` method
2. exposing local/cloud branching to the step

### 3. Agent construction helper

Add a lower-level builder that accepts `ModelConfig` directly.

Recommended helper:

- `build_agent_from_config(config, output_type, system_prompt, tools=None)`

Implementation detail:

- `build_agent_from_args()` remains for CLI entrypoints
- task/step orchestration should move to the config-based builder via `StepRuntime.agent(...)`

### 4. Instance caching

Local runtime instances should be cached by resolved model identity where possible.

Examples:

- `film.shots`: cache key based on detector backend/model-ish identity
- `film.frames`: cache key based on extractor backend and clip model
- shared speech pipeline: cache key composed from STT + diarization + emotion config inputs
- singleton-ish steps with no meaningful model identity may still use a stable fallback key

## Migration Steps

### Phase 1. Introduce runtime facade

- implement `StepRuntime`
- implement `ModelRegistry.for_step(step_id)`
- add internal `_RuntimeSpec` table
- keep existing builder logic private inside the registry

### Phase 2. Add direct config-based agent builder

- add `build_agent_from_config(...)`
- make `build_agent_from_args(...)` delegate to it after resolving precedence

### Phase 3. Migrate agent-backed steps

Update representative steps to call:

- `registry.for_step(step_id).agent(...)`

Targets:

- `novel.scenes`
- `novel.aliases`
- `novel.characters`
- `film.shot_groups`
- `film.scenes`
- `cross.synthesize`

### Phase 4. Migrate instance-backed steps

Update representative steps to call:

- `registry.for_step(step_id).instance()`

Targets:

- `film.shots`
- `film.frames`
- `film.audio.separate`
- `film.audio.vad`
- `film.audio.diarize`
- `film.audio.stt`
- `film.audio.emotion`
- `film.audio.sound_events`
- `film.audio.music`

### Phase 5. Remove step-facing legacy registry API

After migration:

- delete or stop using public `cloud_config()`
- delete or stop using public `local_config()`
- delete or stop using bespoke `get_*()` accessors

The registry should keep only the generic step-runtime entrypoint.

## Error Handling

Capability mismatch errors should be explicit.

Examples:

- calling `.agent()` on `film.shots` should raise a `TypeError` naming `film.shots`
- calling `.instance()` on `novel.aliases` should raise a `TypeError` naming `novel.aliases`

This keeps failures actionable when a step is wired to the wrong runtime capability.

## Testing Plan

### Unit tests for runtime dispatch

Add focused registry tests for:

- `for_step("novel.aliases").agent(...)` dispatches through resolved config
- `for_step("film.shots").instance()` returns the built local object
- wrong capability calls raise clear `TypeError`s

### Unit tests for caching

Add tests for:

- repeated `instance()` calls on the same step reuse the cached object
- speech-pipeline-backed steps reuse the same shared runtime when config identity is the same
- generic `for_step(...)` path exercises the registry without bespoke getters

### Migration coverage

Preserve or update step tests so they still verify:

- non-model step config like `batch_tokens` and `chunk_size` still propagates correctly
- representative cloud and local steps remain constructible through the new registry API

### Config semantics

Verify the redesign preserves:

- step overrides beating model defaults
- existing `project.yaml` model references resolving the same way as before

## Non-Goals

- broader orchestration cleanup
- changing project.yaml schema
- introducing a large capability taxonomy beyond current needs
- redesigning every agent wrapper class unless needed for direct runtime migration

## Recommended End State

- steps ask only for `registry.for_step(step_id)`
- steps choose interaction style via `.agent(...)` or `.instance()`
- local-vs-remote stays internal to the registry
- adding a new step requires a new spec entry, not a new public registry method
