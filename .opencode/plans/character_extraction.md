# Plan B: 角色提取 (Character Extraction)

> **目标**: 从场景文档提取角色级笔记 | **输入**: scene_metadata.json + scene_XXX.txt | **输出**: character_notes.json + character_alias_map.json

---

## 1. 架构概览

```
scene_metadata.json + scene_XXX.txt (from any source)
    │
    ▼
┌──────────────────────────────────────────────┐
│  Step 1: 角色识别 Agent (LLM-based)          │
│  输入: 全文场景集合                           │
│  输出: character_alias_map.json               │
│  目的: 识别角色所有别名形式                   │
└──────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────┐
│  Step 2: 角色级提取 Agent (LLM-based)        │
│  输入: 单个场景 + alias_map + target_chars    │
│  输出: character_notes.json                   │
│  目的: 按角色×场景提取结构化信息              │
└──────────────────────────────────────────────┘
```

**设计原则**: 跨模态通用。无论场景来自小说、影片还是设定集，只要符合统一场景格式，本Pipeline均可处理。

---

## 2. 输入格式规范

### 2.1 统一的Scene输入格式

Plan B期望所有上游Pipeline（小说/影片/设定集）输出统一格式的场景文档：

**scene_metadata.json**:
```json
{
  "metadata": {
    "version": "1.0",
    "source_type": "novel" | "film" | "artbook",
    "project": "超时空辉夜姬"
  },
  "scenes": [
    {
      "scene_id": "c01_s01",
      "source": {
        "type": "novel",
        "file": "超时空辉夜姬.txt"
      },
      "content": {
        "file_path": "scene_c01_s01.txt",
        "text_content": "[可选，直接包含文本]",
        "token_estimate": 1867
      },
      "scene_info": {
        "location": "地球-某城市-街头",
        "time_of_day": "傍晚",
        "timeline_position": "arc_1_chapter_1"
      },
      "characters": {
        "present": ["辉夜"],
        "mentioned": [],
        "pov_character": "辉夜"
      }
    }
  ]
}
```

**scene_XXX.txt** (文本文件):
```
# scene_c01_s01.txt
# Source: 超时空辉夜姬.txt
# Scene ID: c01_s01
# Location: 地球-某城市-街头
# Characters: 辉夜
# ---

[场景内容文本]
```

### 2.2 material.yaml中的角色配置

```yaml
target_characters:
  - name: "辉夜"
    notes: "主要角色，初期人格"
    composite_identity: false
    
  - name: "八千代"
    notes: "辉夜8000年后的形态"
    composite_identity: true
    linked_to: "辉夜"
    
  - name: "彩叶"
    notes: "另一主角"
    composite_identity: false
```

---

## 3. Step 1: 角色识别 Agent

### 3.1 目标
识别每个目标角色在原文中出现的**所有名词形式**（别名）。这是角色提取的前置步骤，解决"辉夜"可能被称为"那孩子"/"公主大人"等问题。

### 3.2 输入
```json
{
  "target_characters": ["辉夜", "八千代", "彩叶"],
  "scenes": [
    {
      "scene_id": "c01_s01",
      "text": "[场景完整文本]"
    }
  ],
  "full_text": "[可选，完整小说文本用于全文分析]"
}
```

### 3.3 输出: character_alias_map.json

```json
{
  "version": "1.0",
  "generated_at": "2026-03-26T10:00:00Z",
  "project": "超时空辉夜姬",
  "alias_map": {
    "辉夜": {
      "primary_name": "辉夜",
      "aliases": [
        "辉夜",
        "那孩子",
        "公主大人",
        "月之民的女孩",
        "黑发少女",
        "来自月球的她"
      ],
      "alias_contexts": {
        "辉夜": ["c01_s01", "c01_s02"],
        "那孩子": ["c01_s03", "c02_s01"],
        "公主大人": ["c03_s02"]
      }
    },
    "八千代": {
      "primary_name": "八千代",
      "aliases": [
        "八千代",
        "AI歌姬",
        "月读空间的管理者",
        "那道身影"
      ],
      "alias_contexts": { ... }
    },
    "彩叶": {
      "primary_name": "彩叶",
      "aliases": ["彩叶", "她", "那个女孩"],
      "alias_contexts": { ... }
    }
  }
}
```

**注意**: 不包含confidence_scores，置信度分层和review记录放在`character_alias_reviewed.json`中。

### 3.4 Prompt模板

```
你是一个专业的文学分析助手。请分析以下文本，识别角色"{character_name}"在文中出现的所有不同称呼方式。

任务要求：
1. 列出该角色在文中被称呼的所有形式（包括名字、代称、描述性称谓等）
2. 对每个别名给出置信度分层："high"或"low"
   - high: 非常确定指向该角色（如直接名称、明确的关系称谓）
   - low: 可能指向该角色，但需人工review确认（如模糊代词、描述性称谓）
3. 标注每个别名出现的场景ID和具体上下文片段
4. 注意排除其他角色的同名或相似称谓
5. 返回JSON格式

示例：
输入角色"辉夜"，可能返回：
- "辉夜"（high - 直接称呼）
- "那孩子"（high - 从彩叶视角明确指辉夜）
- "公主大人"（high - 特定场景中的敬称）
- "黑发少女"（low - 描述性称呼，需确认是否指辉夜）

但不应包含：
- "彩叶"（另一个角色）
- 代词"她"（除非上下文100%确定）

场景文本:
{scene_texts}

目标角色: {character_name}
```

### 3.5 别名置信度分层

LLM只输出两档置信度，不输出具体分数：

| 置信度 | 别名类型 | 示例 | 后续处理 |
|--------|---------|------|---------|
| **high** | 直接名称、明确关系称谓 | "辉夜", "那孩子"(已知关系) | 直接收录，人工抽查 |
| **low** | 描述性称谓、模糊代词 | "黑发少女", "她" | 必须人工review确认 |

**说明**:
- LLM的精确confidence score校准性不好，不输出
- 用high/low两档足够区分"明显是"vs"可能是"
- 人工review结果持久化到`character_alias_reviewed.json`保证可复现

### 3.6 人工Review文档

**文件**: `character_alias_reviewed.json`

目的：
- 持久化人工review决策，保证可复现
- 记录每个别名判断的依据和上下文
- 支持同一别名在不同场景指不同角色的情况

**格式**:
```json
{
  "version": "1.0",
  "reviewed_at": "2026-03-26T15:00:00Z",
  "reviewer": "human",
  "decisions": [
    {
      "character": "辉夜",
      "alias": "那孩子",
      "context": {
        "scene_id": "c01_s03",
        "chapter": "ch003",
        "excerpt": "...那孩子站在街头，困惑地看着四周..."
      },
      "llm_confidence": "high",
      "decision": "confirmed",
      "review_notes": "从上下文'来自月球'、'黑发'明确指向辉夜"
    },
    {
      "character": "辉夜",
      "alias": "那孩子",
      "context": {
        "scene_id": "c10_s05",
        "chapter": "ch010"
      },
      "llm_confidence": "low",
      "decision": "rejected",
      "review_notes": "这里'那孩子'指彩叶回忆中的童年朋友，非辉夜"
    }
  ]
}
```

**关键设计**:
- 按**场景**记录别名判断，非全局
- 支持同一别名在不同场景的不同判定
- `llm_confidence`: LLM最初判断的high/low
- `decision`: confirmed(确认)/rejected(拒绝)/uncertain(待定)
- `context.excerpt`: 关键上下文片段，便于review

---

## 4. Step 2: 角色级提取 Agent

### 4.1 目标
对每个目标角色的每个场景，提取结构化角色笔记。这是yorishiro.md §2.4定义的核心提取逻辑。

### 4.2 输入
```json
{
  "character_name": "辉夜",
  "scene": {
    "scene_id": "c01_s01",
    "text": "[场景完整文本]",
    "metadata": {
      "location": "地球-某城市-街头",
      "time_of_day": "傍晚",
      "timeline_position": "arc_1_chapter_1"
    }
  },
  "alias_map": {
    "辉夜": ["辉夜", "那孩子", "公主大人"]
  },
  "other_characters": ["彩叶", "八千代"]
}
```

### 4.3 输出: character_notes.json

```json
{
  "metadata": {
    "version": "1.0",
    "generated_at": "2026-03-26T10:00:00Z",
    "project": "超时空辉夜姬",
    "source_type": "novel",
    "total_notes": 312,
    "characters": ["辉夜", "八千代", "彩叶"]
  },
  "notes": [
    {
      "note_id": "kaguya_c01_s01",
      "character": "辉夜",
      "scene_id": "c01_s01",
      "source": {
        "type": "novel",
        "file": "超时空辉夜姬.txt"
      },
      "timeline": {
        "position": "arc_1_chapter_1",
        "relative_time": "故事开始"
      },
      
      "presence": {
        "is_present": true,
        "role_in_scene": " protagonist"
      },
      
      "dialogue": {
        "samples": [
          "这是什么地方？",
          "我...我不是故意的"
        ],
        "language_traits": "使用疑问句较多，语气不确定，带有天真和困惑"
      },
      
      "emotion_motivation": {
        "emotional_state": "困惑、好奇、轻微不安",
        "inferred_motivation": "试图理解新环境，寻找回到月球的方法",
        "internal_conflict": "对地球的陌生感 vs 探索欲望"
      },
      
      "relationships": [
        {
          "target": "彩叶",
          "dynamic": "初次相遇，保持警惕但好奇",
          "shift": "新关系建立"
        }
      ],
      
      "behavior": {
        "actions_taken": ["观察周围环境", "尝试使用能力", "与彩叶对话"],
        "actions_avoided": ["没有直接攻击人类", "没有显露全部能力"],
        "decision_logic": "选择隐藏身份，先观察再行动"
      },
      
      "arc": {
        "is_arc_marker": false,
        "arc_phase": "起点-适应期",
        "notes": "角色初期的典型表现"
      },
      
      "knowledge": {
        "facts_revealed": ["来自月球", "拥有特殊能力"],
        "facts_hidden": ["真实身份", "来地球的目的"],
        "misconceptions": ["以为地球人类都很原始"]
      }
    }
  ]
}
```

### 4.4 数据结构 Schema

参考 yorishiro.md §2.4，但简化部分字段：

```python
class CharacterNote(BaseModel):
    note_id: str  # {character}_{scene_id}
    character: str
    scene_id: str
    source: dict
    
    timeline: dict
    presence: dict
    dialogue: dict
    emotion_motivation: dict
    relationships: List[dict]
    behavior: dict
    arc: dict
    knowledge: dict
    
    # 复合身份角色专用
    active_persona: Optional[str]  # "辉夜" or "八千代"
```

### 4.5 Prompt模板

```
你是一个专业的角色分析师。给定以下场景内容和角色别名信息，请针对角色"{character_name}"提取结构化信息。

角色别名映射:
{alias_map}

场景内容:
{scene_text}

场景元数据:
- 地点: {location}
- 时间: {time_of_day}
- 时间线位置: {timeline_position}
- 其他在场角色: {other_characters}

提取要求（JSON格式）:
1. presence: 角色是否在场，在场景中的角色定位
2. dialogue: 
   - samples: 该角色的原文台词（保留原文，不要改写）
   - language_traits: 观察到的语言特征（口头禅/句式/语域）
3. emotion_motivation:
   - emotional_state: 当前情绪状态
   - inferred_motivation: 推断的行为动机
   - internal_conflict: 如有内心冲突，描述之
4. relationships: 与其他角色的互动动态
5. behavior:
   - actions_taken: 做了什么
   - actions_avoided: 有机会做但选择不做的事（重要！）
   - decision_logic: 决策背后的逻辑推断
6. arc: 是否为角色发展的关键节点
7. knowledge: 角色在本场景中获知/隐瞒的信息

如果该角色未在场景中出现，只输出: {"presence": {"is_present": false}}
```

---

## 5. 数据格式规范

### 5.1 输出目录结构

```
output/
├── novel/                     # Plan A输出
│   ├── scene_metadata.json
│   └── scene_XXX.txt
├── character/                 # Plan B输出
│   ├── character_alias_map.json
│   └── character_notes.json
```

### 5.2 character_alias_map.json Schema

```python
class CharacterAlias(BaseModel):
    primary_name: str
    aliases: List[str]
    alias_contexts: Dict[str, List[str]]  # alias -> scene_ids

class AliasMap(BaseModel):
    version: str
    generated_at: str
    project: str
    alias_map: Dict[str, CharacterAlias]
```

**注意**: 不包含confidence_scores，置信度分层和review记录在`character_alias_reviewed.json`中。

### 5.3 character_notes.json Schema

```python
class CharacterNotes(BaseModel):
    metadata: dict
    notes: List[CharacterNote]  # 见§4.4
```

### 5.4 character_alias_reviewed.json Schema

```python
class AliasContext(BaseModel):
    scene_id: str
    chapter: str
    excerpt: str  # 关键上下文片段

class ReviewDecision(BaseModel):
    character: str
    alias: str
    context: AliasContext
    llm_confidence: str  # "high" | "low"
    decision: str        # "confirmed" | "rejected" | "uncertain"
    review_notes: str

class ReviewedAliasMap(BaseModel):
    version: str
    reviewed_at: str
    reviewer: str
    decisions: List[ReviewDecision]
```

---

## 6. 实现计划

### 6.1 框架选择

**PydanticAI + LiteLLM**（同 Plan A）

- **强制工具输出**: 所有 Agent 使用 `result_type` 指定 Pydantic 模型，强制结构化输出
- **类型安全**: 通过 Pydantic 验证输入输出格式
- **无需手动 JSON 解析**: 直接操作类型化的 Python 对象

```python
# 角色识别 Agent 示例
class AliasRecognitionResult(BaseModel):
    aliases: List[AliasCandidate] = Field(description="识别到的所有别名候选")
    
class AliasCandidate(BaseModel):
    alias: str = Field(description="别名文本")
    confidence_tier: str = Field(description="置信度分层: high 或 low")
    scene_id: str = Field(description="出现的场景ID")
    excerpt: str = Field(description="上下文片段")

alias_agent = Agent(
    model="claude-3-5-sonnet",
    result_type=AliasRecognitionResult,
    system_prompt="识别角色在文本中的所有称呼方式..."
)
```

### 6.2 模块结构

```
yorishiro/
├── agents/
│   └── character/
│       ├── __init__.py
│       ├── alias_recognition.py    # Step 1: 角色识别Agent (强制工具输出)
│       └── character_extraction.py # Step 2: 角色级提取Agent (强制工具输出)
├── pipelines/
│   └── character_extraction.py     # Pipeline主控
├── models/
│   └── character_models.py         # Pydantic模型
└── utils/
    └── alias_matcher.py            # 别名匹配工具
```

### 6.2 开发任务分解

| 任务 | 描述 | 优先级 | 预估工时 |
|------|------|--------|----------|
| 1 | 定义统一Scene输入接口 | 高 | 2h |
| 2 | 设计角色识别Agent | 高 | 4h |
| 3 | 设计角色级提取Agent | 高 | 6h |
| 4 | 实现别名匹配工具 | 中 | 3h |
| 5 | 设计数据模型(Pydantic) | 高 | 3h |
| 6 | 实现Pipeline主控逻辑 | 高 | 4h |
| 7 | 编写测试用例 | 中 | 4h |
| 8 | 集成测试 | 高 | 3h |

**总计**: ~29工时

---

## 7. 跨模态通用性设计

### 7.1 不同来源的场景适配

| 来源 | 文本来源 | 特殊处理 |
|------|---------|---------|
| 小说 | scene_XXX.txt | 直接使用 |
| 影片 | transcript.txt + 关键帧描述 | 转录文本 + VLM描述 |
| 设定集 | entry文本 | 结构化条目 |

### 7.2 Source Type标记

每个note记录source_type，用于后续处理：

```json
{
  "source": {
    "type": "novel" | "film" | "artbook",
    "file": "...",
    "details": {
      // 模态特定的额外信息
    }
  }
}
```

---

## 8. 与上下游的接口

### 8.1 上游输入 (Plan A或其他)

必须是统一Scene格式，详见§2.1。

### 8.2 下游输出 (Plan C: SOUL.md合成)

character_notes.json 直接作为Synthesis Agent的输入。

---

## 9. 已确认设计决策

### 9.1 别名更新策略 ✓
- **无需特别策略**: alias_map是角色提取的前置依赖，变化后自然重跑对应角色的提取
- **review文档持久化**: 人工review结果写入`character_alias_reviewed.json`，保证可复现

### 9.2 增量提取策略 ✓
- **是，只提取新增场景**: 记录已处理scene_id，跳过已存在的
- **别名识别**: 新增场景时只识别其中的新别名实例

### 9.3 多角色策略 ✓
- **按需逐个提取**: 不默认提取全部角色，按material.yaml中指定的目标角色逐个处理
- **非并行vs串行问题**: 角色数量通常<10，串行处理足够，无需复杂并行逻辑

### 9.4 置信度策略 ✓
- **分层**: LLM只输出"high"/"low"两档，不输出精确分数
- **high**: 直接收录，人工抽查
- **low**: 必须人工review确认
- **review记录**: 所有low置信度别名及review决策写入`character_alias_reviewed.json`

---

**下一步**: 开始实现角色识别Agent (Step 1)。
