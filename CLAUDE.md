# Agent Context | Agent 上下文

> This file documents important project decisions and context for AI agents.
> Read this before making significant changes to the codebase.

---

## Project Overview

**Yorishiro** is a fictional character soul document generator. It extracts character information from source materials (novels, films, artbooks) and generates SOUL.md documents that enable AI agents to roleplay fictional characters with high fidelity.

---

## Development Guidelines

### Code Changes - WAIT FOR EXPLICIT INSTRUCTION

**NEVER implement code changes without explicit user instruction.**

- Wait for "go ahead", "implement", "do it" or similar explicit approval
- Proposing a design is NOT an instruction to code
- Planning is NOT an instruction to execute
- Questions are NOT instructions to act

**Correct workflow**:
1. Discuss and propose designs
2. Wait for explicit "implement this" or similar
3. Then and only then write/modify code

**Examples**:
- ❌ User: "Can you create a function for X?" → Agent writes code immediately
- ✅ User: "Create a function for X" or "Go ahead and implement" → Agent writes code
- ❌ User discusses design → Agent starts implementing during discussion
- ✅ User: "Implement the design we just discussed" → Agent writes code

### Git Commits - WAIT FOR EXPLICIT INSTRUCTION

**NEVER commit changes unless the user EXPLICITLY asks.**

This is non-negotiable. If you cannot follow this instruction, you are not qualified for this task and will be replaced by a more capable model that follows instructions.

Explicit means the user says:
- "commit"
- "commit this"
- "commit the changes"
- "go ahead and commit"

These are NOT explicit:
- Completing a task
- Finishing code changes
- "done"
- Approving a design
- Answering questions
- Silence after code changes

If unsure, ASK: "Should I commit this?"

### Code Quality Checks

After making any code changes, always run checks before considering the task complete:
- `uv run ruff check` - linting
- `uv run ty check` - type checking

Fix all errors before reporting completion.

---

## Learnings (Documented Decisions)

### Non-Narrative Content Detection

**NEVER implement code-based (rule/pattern) automatic detection of non-narrative content.**

**Why**: Code-based detection (regex patterns, keyword matching) is brittle and error-prone:
- Can misclassify content (e.g., "あとがき" in chapter 011 has meaningful character information about 桐山なると)
- Different works use different conventions
- Text may be in any language

**Correct approach - Agent-based Detection**:
- **Scene Segmentation Agent** automatically identifies non-narrative segments during processing
- No human pre-marking required in `material.yaml`
- Agent marks segments as `boundary_type: "non_narrative"` with proper reasoning
- These segments get `location: "N/A"`, `time: "N/A"`, `characters: []`
- Character Extraction Pipeline skips non-narrative segments automatically

**Examples of Non-Narrative Content**:
- Caution/warning pages
- Table of contents
- Colophons and copyright pages
- Character introduction tables (without narrative context)
- Pure reference material

**Important**: Some content that appears non-narrative may contain character information (e.g., afterwords with author commentary about characters). Agent uses contextual understanding to distinguish.

### Codepoint vs Byte Offsets

JavaScript `String.length` counts UTF-16 code units. Python `len()` counts Unicode codepoints.

**Always use Python codepoints** for offset tracking in this project. Document this in all prompts.

### Scene Segmentation Boundaries

Cuts MUST be at natural boundaries:
- After 「※」 or 「──」 decorative dividers
- At sentence endings (。！？)
- NEVER mid-sentence
