**注：施工现场中👷 | Note: Work In Progress**

# Yorishiro（依り代）

### Fictional Character Soul Document Generator & Chat System
### 虚构角色灵魂文档生成与对话系统

---

## Overview | 概览

Yorishiro is a system that extracts character information from fictional works (films, novels, artbooks, interviews) and generates **SOUL.md** — a character soul document that enables AI agents to roleplay with high fidelity and low OOC (Out-of-Character) rate.

Yorishiro 是一个从虚构作品（影视、小说、设定集、访谈等）中提取角色信息并生成 **SOUL.md** 灵魂文档的系统，使 AI Agent 能够以角色人格进行高保真、低 OOC（出戏）率的对话。

### What is SOUL.md? | 什么是 SOUL.md?

SOUL.md is a structured character persona document containing:

SOUL.md 是一份结构化的角色人格文档，包含：

- **Core Identity** — 核心身份与驱动力
- **Personality Model** — 人格模型（价值观、动机、恐惧、认知模式）
- **Voice & Language** — 语言风格（口头禅、句式、情绪表达）
- **Relationships** — 关系图谱
- **Behavioral Patterns** — 行为模式
- **Negative Constraints** — 负面约束（角色"绝不会做"的事）
- **Character Arc** — 角色弧线演变
- **World Knowledge** — 角色知道/不知道的信息边界

---

## Architecture | 架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        Source Materials                          │
│            Film ─── Novel ─── Artbook ─── Interviews             │
└──────┬──────────────┬──────────────┬──────────────┬─────────────┘
       │              │              │              │
       ▼              ▼              ▼              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Phase 1: Extraction Layer                        │
│    Scene Detection → Keyframe/Text Extraction → Character Notes   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Phase 2: Index Layer                            │
│              Vector Database + Metadata → Retrieval API           │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Phase 3: Alignment Layer                         │
│              Cross-Source Fuzzy Matching (Film ↔ Novel)          │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Phase 4: Synthesis Layer                        │
│          SOUL.md Generation via Agent + Consistency Check         │
└──────────┬───────────────────────────────────────────────────────┘
           │
           ▼
      [ SOUL.md Output ]
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Phase 5: Runtime Distribution                    │
│                                                                  │
│   ┌────────────────────┐    ┌────────────────────────────┐      │
│   │  SOUL.md           │    │  Yorishiro MCP Server      │      │
│   │  (System Prompt)   │    │  (Memory Tools)            │      │
│   └────────────────────┘    └────────────────────────────┘      │
│                                                                  │
│   MCP Hosts: Claude Desktop / Cursor / Claude Code / Custom       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Features | 核心特性

| Feature | Description | 说明 |
|---------|-------------|------|
| **Multi-Source Extraction** | Process film, novel, artbook, and interview materials | 多源素材处理（影视/小说/设定集/访谈） |
| **Split-Persona Support** | Handle characters with multiple identities (e.g., past/present self) | 复合身份角色支持（如过去/现在双重人格） |
| **Knowledge Boundaries** | Filter information by what the character should know | 角色知识边界过滤 |
| **Swappable Models** | Use any LLM/embedding provider via config | 模型可替换（通过配置切换不同 provider） |
| **MCP Integration** | Expose memory tools via Model Context Protocol | MCP 协议集成，暴露记忆工具 |

---

## Dual-Layer Memory Architecture | 双层记忆架构

| Layer | Purpose | Timing | Usage |
|-------|---------|--------|-------|
| **SOUL.md** | Long-term memory / Persona core | Static | Loaded into system prompt |
| **Memory System** | Situational memory / Encyclopedia | Runtime | On-demand RAG retrieval |

This design ensures:
- Fast character responses via SOUL.md in system prompt
- Detailed scene/dialogue/world knowledge via runtime retrieval

---

## Current Status | 当前状态

| Item | Status |
|------|--------|
| **Version** | 0.4 |
| **State** | Draft |
| **Phase** | Phase 1 (Extraction — all steps complete, pipeline orchestration in progress) |

---

## Project Structure | 项目结构

### Code | 代码

```
yorishiro/
├── tasks/                   # Phase 1: Task/Step abstractions per pipeline step
│   ├── base.py              #   Task + Step ABCs
│   ├── registry.py          #   ModelRegistry (lazy-loads local models + resolves cloud configs)
│   ├── novel/               #   Novel pipeline steps (chapters, scenes, aliases, characters)
│   ├── film/                #   Film pipeline steps (shots, frames, audio, shot_groups, scenes)
│   └── cross/               #   Cross-source steps (synthesize)
├── pipeline/
│   └── orchestrator.py      #   Dependency resolution + execution + backup
├── video/                   #   Local ML: shot detection, keyframe extraction
├── audio/                   #   Local ML: speech, diarization, sound events, music
├── agents/                  #   LLM/VLM agents (shot grouping, scene analysis)
├── models/                  #   Pydantic data models
├── project.py               #   Project config loader + path management
├── agent_utils.py           #   Shared pydantic-ai agent builder
├── backup.py                #   Hard-link snapshot backup
└── cli.py                   #   Unified CLI entry point
```

### Project Folder | 项目文件夹

```
projects/{CODE}/
├── project.yaml             # Config: sources, models, steps, step_groups
├── raw/                     # Read-only source inputs (epub, mkv, …)
├── processed/
│   └── {source-id}/
│       └── steps/
│           ├── chapters/    # novel.chapters
│           ├── scenes/      # novel.scenes  (per-chapter subdirs)
│           ├── aliases/     # novel.aliases
│           ├── characters/  # novel.characters  (per-character subdirs)
│           ├── shots/       # film.shots
│           ├── frames/      # film.frames
│           ├── audio/       # film.audio
│           ├── shot_groups/ # film.shot_groups
│           └── scenes/      # film.scenes
├── cross/                   # Cross-source intermediates
└── souls/                   # Final SOUL.md outputs
```

---

## Usage | 使用

```bash
# Run a single step
python -m yorishiro run --project projects/CPK --source cpk-novel --step novel.scenes

# Run a single task within a step (e.g. one chapter)
python -m yorishiro run --project projects/CPK --source cpk-novel --step novel.scenes --task ch003

# Run a named step group
python -m yorishiro run --project projects/CPK --source cpk-novel --group novel-full

# Run all steps for all sources
python -m yorishiro run --project projects/CPK --all

# Force re-run (ignore staleness)
python -m yorishiro run --project projects/CPK --source cpk-film --step film.audio --force

# Check completion status
python -m yorishiro status --project projects/CPK
```

---

## Documentation | 文档

| Document | Description |
|----------|-------------|
| [docs/REQUIREMENTS.md](./docs/REQUIREMENTS.md) | Product goals, SOUL.md format spec, roadmap (Chinese) |
| [docs/DESIGN.md](./docs/DESIGN.md) | Technical design: pipeline architecture, tool choices, schemas, prompt templates (Chinese) |
| [docs/PLAN.md](./docs/PLAN.md) | Implementation status, next tasks, open questions |

## Examples | 示例

| Example | Description |
|---------|-------------|
| [BambooCutter](./examples/BambooCutter/) | 竹取物语 (Tale of the Bamboo Cutter) example project |

---

## Model Configuration | 模型配置

Yorishiro supports swappable model providers via `project.yaml`. Models are defined once by name and referenced from step configs:

```yaml
models:
  sonnet:                          # cloud model
    provider: openrouter
    name: anthropic/claude-sonnet-4-6
    thinking: medium
    output_mode: tool
    api_key_env: YORISHIRO_API_KEY_OPENROUTER

  whisper:                         # local model
    backend: faster-whisper
    model: large-v3
    device: auto

steps:
  novel.scenes:
    model: sonnet
    batch_tokens: 32000
  film.audio:
    model: whisper
    diarization_model: diarization
  cross.synthesize:
    model: sonnet
    thinking: high                 # step-level override
```

Supported cloud providers: OpenRouter (Claude, GPT, Gemini, …), OpenAI.  
Supported local backends: faster-whisper, pyannote, CLIP, adaptive shot detector, Demucs.

---

## License

MIT License
