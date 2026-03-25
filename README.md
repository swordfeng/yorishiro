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
| **Version** | 0.3 |
| **State** | Draft |
| **Phase** | Phase 0 (Core Hypothesis Validation) |

For full architecture details, see [yorishiro.md](./yorishiro.md).

---

## Project Structure | 项目结构

```
yorishiro/
├── extract/                 # Phase 1: Extraction pipelines
│   ├── film_pipeline.py    #   Film processing (scene detection, keyframes, STT)
│   ├── novel_pipeline.py   #   Novel processing (segmentation, summarization)
│   └── character_extractor.py  # Character-centric extraction
├── index/                  # Phase 2: Vector storage & retrieval
│   ├── indexer.py          #   Vector database operations
│   └── searcher.py         #   Retrieval API
├── align/                  # Phase 3: Cross-source alignment
│   └── cross_source_aligner.py
├── synthesize/             # Phase 4: SOUL.md synthesis
│   ├── agent.py            #   Synthesis agent
│   └── templates.py        #   SOUL.md templates
├── mcp_server/            # Phase 5: MCP server implementation
│   ├── server.py          #   MCP server entry point
│   ├── tools.py           #   MCP tool definitions
│   └── knowledge_gate.py  #   Knowledge boundary filtering
├── config.yaml             # Model provider configuration
└── cli.py                 # CLI entry point
```

---

## Documentation | 文档

| Document | Description |
|----------|-------------|
| [yorishiro.md](./yorishiro.md) | Full architecture document (Chinese) |
| [SOUL.md Format](./yorishiro.md#_5) | SOUL.md specification |
| [MCP Integration](./yorishiro.md#_65) | MCP server usage |

---

## Model Configuration | 模型配置

Yorishiro supports swappable model providers via `config.yaml`:

```yaml
models:
  extraction:     # Smaller models for extraction tasks
    provider: "openrouter"  # or "local"
    model: "anthropic/claude-3-haiku"
    
  synthesis:      # Strong models for SOUL.md synthesis
    provider: "openrouter"
    model: "anthropic/claude-3-opus"
    
  embedding:      # Vector embeddings
    provider: "openai"
    model: "text-embedding-3-large"
```

Supported providers:
- **OpenRouter** — Claude, GPT, Gemini, and 100+ other models
- **Local Models** — Ollama, Qwen, and other local LLMs
- **OpenAI** — GPT embeddings
- **Local Embeddings** — BGE-M3, CLIP

---

## License

MIT License
