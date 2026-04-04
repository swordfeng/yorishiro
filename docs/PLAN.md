# Yorishiro — Project Plan

> **Version**: 0.4 · **State**: Active · **Last updated**: 2026-04-03

---

## Current Status

| Step | Module | Status |
|------|--------|--------|
| `novel.chapters` | `chapters_epub.py` | ✅ Done |
| `novel.scenes` | `scene.py` | ✅ Done |
| `novel.aliases` | `aliases.py` | ✅ Done |
| `novel.characters` | `character.py` | ✅ Done |
| `cross.synthesize` | `synthesize.py` | ✅ Done |
| `film.shots` | `video/shot_detector.py` | ✅ Done |
| `film.frames` | `video/keyframe_extractor.py` | ✅ Done |
| `film.audio` | `audio/speech_pipeline.py` + others | ✅ Done |
| `film.shot_groups` | `agents/film/shot_grouping.py` | ✅ Done |
| `film.scenes` | `agents/film/scene_analysis.py` | ✅ Done |
| **Pipeline orchestration** | `tasks/` + `pipeline/` + `cli.py` | 🔲 In progress |
| Phase 2: Index Layer | — | 🔲 Not started |
| Phase 3: Alignment Layer | — | 🔲 Not started |
| Phase 5: MCP Server | — | 🔲 Not started |

---

## Immediate Next Tasks

1. **Finish pipeline orchestration** (current sprint):
   - `yorishiro/tasks/base.py` — `Task` + `Step` ABCs
   - `yorishiro/tasks/registry.py` — `ModelRegistry`
   - `yorishiro/tasks/novel/` — thin wrappers over existing domain modules
   - `yorishiro/tasks/film/` — thin wrappers over existing domain modules
   - `yorishiro/tasks/cross/synthesize.py`
   - `yorishiro/pipeline/orchestrator.py`
   - `yorishiro/cli.py` — unified `python -m yorishiro run ...` entry point
   - Migrate `projects/CPK/project.yaml` to new schema

2. **Phase 2 — Index Layer**:
   - Design vector DB schema (Qdrant collections)
   - Indexer for novel scenes + character notes
   - Indexer for film scene content + keyframes

3. **Phase 3 — Alignment**:
   - Cross-source fuzzy matching (film ↔ novel scene pairing)
   - Alignment map with confidence scores

---

## Open Questions

| # | Question | Decision needed |
|---|----------|----------------|
| 1 | Timeline granularity for knowledge boundary filtering | Before Phase 2 |
| 2 | Multi-language strategy (ja + zh sources for same character) | Before cross.synthesize |
| 3 | OOC evaluation methodology | Before Phase 5 |
| 4 | Knowledge scope annotation cost vs quality tradeoff | Before Phase 2 |
| 5 | `cross.synthesize` step: merge strategy when sources conflict | Before implementation |
