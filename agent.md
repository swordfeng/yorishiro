# Agent Context | Agent 上下文

> This file documents important project decisions and context for AI agents.
> Read this before making significant changes to the codebase.

---

## Project Overview

**Yorishiro** is a fictional character soul document generator. It extracts character information from source materials (novels, films, artbooks) and generates SOUL.md documents that enable AI agents to roleplay fictional characters with high fidelity.

---

## Key Decisions Made

### Authority Levels for Source Materials

Source materials are classified by trustworthiness:

| Level | Name | Description | Examples |
|-------|------|-------------|----------|
| PRIMARY | 最高权威 | Original source material | Original novel, script, official artbook |
| SECONDARY | 次级 | Derivative/authorized works | Translations, adaptations |
| TERTIARY | 低权威 | Unverified sources | Fan wikis, internet discussions |

**Metadata file**: `material.yaml` (git-ignored, contains local paths)

**Conflict resolution**: When sources conflict, higher authority sources are preferred.

### No Vector Database in Phase 0

Phase 0 uses simple JSON/YAML files for extracted data. Vector database (Qdrant) is introduced in Phase 1 when:
- Scale increases (multiple sources per character)
- Runtime retrieval is needed
- Latency requirements become critical

---

## Important Conventions

### SOUL.md Format

See `yorishiro.md` Section 5.1 for the full template. Key sections:

1. Core Identity
2. Personality Model
3. Voice & Language (includes dialogue examples in 3.6)
4. Relationships
5. Behavioral Patterns
6. Negative Constraints
7. Character Arc
8. World Knowledge & Knowledge Boundaries
9. Simulation Directives

### Source Metadata

All sources must be registered in `material.yaml` with:
- `path`: relative path to file
- `authority`: PRIMARY / SECONDARY / TERTIARY
- `language`: ISO language code
- `description`: brief description

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

### Character Name Aliases (Coreference Resolution)

Characters may appear under different names in different scenes. This requires a dedicated resolution pass BEFORE character extraction.

**Coreference Resolution Method:**
1. **Read-until-understood**: No language-specific patterns. Read scenes sequentially until you understand who the alias refers to.
2. For each alias, read its first occurrence scene. If unclear, continue reading subsequent scenes.
3. **Never use pattern matching** - text may be in any language. Only rely on contextual understanding.
4. If entire book read and still unclear → flag for manual review.

**Execution Flow:**
```
Step 1: Collect all unique aliases from scenes_manifest.json
Step 2: Process each alias in order of first appearance
        - Read alias's first scene
        - If unclear, continue reading subsequent scenes
        - Continue until resolved or end of book
Step 3: Output character_aliases.json
```

**Output Format:**
```json
{
  "かぐや": [
    {"chapter": "ch004", "scene": 0, "alias": "赤ちゃん"},
    {"chapter": "ch004", "scene": 1, "alias": "少女"}
  ],
  "彩葉": [...],
  "八千代": [...],
  "UNRESOLVED": [...]  // Cannot resolve, needs manual review
}
```

**Important Notes:**
- Canonical names use 日文原文 as they appear in source material
- "叙述者" is excluded from alias mapping (narrator, not a character)
- Channel/duo names (e.g., "いろＰ") may be aliases for characters - read until understood
- Manual confirmation is one-time at the end, not per-chapter

### Manifest Naming

Scene manifests MUST be named `scenes_manifest.json` (not `manifest.json` or other variants).

## Current Test Material

**CPK.epub** - 超かぐや姫！ (Cosmic Princess Kaguya!) Japanese novelization
- Extracted: 16 chapters
- Release: 2026-01-22 (Netflix)
- Studio: Studio Colorido / Studio Chromato

### Key Characters

| Character | Role | Note |
|-----------|------|------|
| 酒寄彩葉 (Iroha Sakayori) | Protagonist | High school student, works part-time, Yachiyo fan |
| 月見ヤチヨ (Yachiyo Runami) | AI singer | Future Kaguya, 8000 years later |
| 輝耀 (Kaguya) | Mysterious girl | Same entity as Yachiyo, different time period |
| フシ (Fushi) | Companion | Evolved from InuDOGE over 8000 years |

### Split Persona Architecture Test Case

This material is **perfect for testing Yorishiro's split-persona design**:
- Yachiyo = Kaguya from the future (same soul, different identity)
- Yachiyo has knowledge Kaguya doesn't (her own future)
- Iroha is the connection point between both personas
- Knowledge boundary filtering is critical here
