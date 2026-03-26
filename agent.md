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

### Phase 0 Strategy

Phase 0 uses **Agent-assisted + manual review** workflow instead of fully manual or fully automated:

1. LLM performs extraction tasks (scene segmentation, character extraction)
2. Human spot-checks results for quality
3. Agent synthesizes SOUL.md
4. Human evaluates OOC rate

**Why**: Balances speed and quality while validating core assumptions.

### Model Configuration

Models are swappable via `config.yaml`:
- Extraction tasks → smaller/cheaper models
- Synthesis tasks → stronger models (Claude Opus recommended)
- Embedding → separate configurable provider

### No Vector Database in Phase 0

Phase 0 uses simple JSON/YAML files for extracted data. Vector database (Qdrant) is introduced in Phase 1 when:
- Scale increases (multiple sources per character)
- Runtime retrieval is needed
- Latency requirements become critical

---

## Current Phase

**Phase 0**: Core hypothesis validation

- [x] Architecture defined in yorishiro.md
- [x] README.md created
- [x] epub parsing pipeline implemented (ebooklib)
- [x] Chapter splitting (16 chapters)
- [x] Scene segmentation (117 scenes across 9 narrative chapters)
- [ ] Character extraction from scenes (彩葉, ヤチヨ, etc.)
- [ ] SOUL.md synthesis
- [ ] SOUL.md quality assessed

---

## File Structure

```
yorishiro/
├── extract/           # Extraction pipelines
│   ├── __init__.py
│   ├── epub_pipeline.py
│   ├── split_chapters.py
│   ├── scene_segmentation.py
│   ├── character_extractor.py
│   ├── generate_prompts.py
│   └── generate_scene_prompts.py
├── cli.py            # CLI entry point
├── main.py           # Main entry point
├── material/          # Source materials
│   ├── raw/          # Original source files
│   │   └── novel/
│   │       ├── CPK.epub
│   │       └── CPK_CN.epub
│   └── processed/     # Processed/extracted content
│       └── novel/CPK/
│           ├── chapters/           # Individual chapter JSON files
│           ├── scenes/            # Scene segmentation
│           │   ├── ch003/scene_XXX.txt (9 scenes)
│           │   ├── ch004/scene_XXX.txt (32 scenes)
│           │   └── ... (117 total scenes)
│           └── character_notes/    # Character extraction (future)
├── material.yaml      # Source material metadata (gitignored)
├── pyproject.toml     # Python project config
├── uv.lock           # Locked dependencies
├── yorishiro.md       # Full architecture document
├── README.md          # Project overview
└── agent.md          # This file - AI agent context
```

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

### Adding New Pipelines

1. Add to `extract/` directory
2. Register in `config.yaml`
3. Update this file if architecture changes

### Modifying SOUL.md Template

Changes to `yorishiro.md` Section 5.1 affect all future generation. Consider:
- Backward compatibility with existing SOUL.md files
- Migration strategy if needed

---

## Open Questions

- [ ] Which character will be used for Phase 0 validation? (Iroha Sakayori recommended as protagonist)
- [ ] OOC evaluation methodology to be defined
- [x] epub parsing library choice: ebooklib (works with CPK.epub)

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

### Story Summary

Iroha Sakayori, a high school student living alone in Tokyo, discovers a baby inside a glowing utility pole. The baby claims to be from the Moon and names herself Kaguya. Together, they enter the Yachiyo Cup streaming tournament. When Kaguya is taken away by lunar beings, Iroha completes a song her late father started. This song echoes through time and reaches Yachiyo — revealing that Yachiyo is Kaguya from 8000 years in the future, who uploaded her consciousness into the virtual world Tsukuyomi.
