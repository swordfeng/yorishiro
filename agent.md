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
- [ ] epub parsing pipeline implemented
- [ ] Phase 0 workflow validated with sample character
- [ ] SOUL.md quality assessed

---

## File Structure

```
yorishiro/
├── material/          # Raw source materials (gitignored)
│   ├── novel/
│   └── film/
├── material.yaml      # Source material metadata (gitignored)
├── pyproject.toml     # Python project config
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

- [ ] Which character will be used for Phase 0 validation?
- [ ] OOC evaluation methodology to be defined
- [ ] epub parsing library choice (python-calibre or epub2py?)
