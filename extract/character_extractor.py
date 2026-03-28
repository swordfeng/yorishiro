"""Character extraction prompt templates and Pydantic models for LLM extraction."""

from __future__ import annotations

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Pydantic models for structured character extraction output
# ---------------------------------------------------------------------------


class Relationship(BaseModel):
    target: str
    dynamic: str
    shift: str | None = None


class KnowledgeScope(BaseModel):
    facts_revealed: list[str]
    facts_hidden: list[str]


class CharacterSceneNote(BaseModel):
    """Structured extraction of one character's information in one scene."""

    character: str
    chapter_index: int
    scene_index: int
    source: str = "novel"
    active_persona: str
    dialogue_samples: list[str]
    language_traits: str
    emotional_state: str
    inferred_motivation: str
    internal_conflict: str
    relationships: list[Relationship]
    actions_taken: str
    actions_avoided: str
    decision_logic: str
    arc_marker: str
    knowledge_scope: KnowledgeScope


class SceneExtractionResult(BaseModel):
    """Multi-character extraction result for a single scene.

    notes contains one entry per target character that actually appears in the scene.
    Characters not present in the scene are omitted entirely.
    """

    notes: list[CharacterSceneNote]


# ---------------------------------------------------------------------------
# Multi-character per-scene prompt (used by extract.character)
# ---------------------------------------------------------------------------

SCENE_SYSTEM_PROMPT = """你是一个专业的角色分析师。你的任务是从给定的小说场景中，同时提取多个目标角色的结构化信息。

## 输出要求

输出 JSON 格式的 SceneExtractionResult，包含 notes 字段（数组）。
每个 note 对应一个在本场景中实际出现的目标角色。
**未出现在本场景中的角色，不要包含在 notes 中。**

每个 note 的结构：

```json
{
  "character": "正式角色名（使用任务中提供的 canonical_name）",
  "chapter_index": 章节索引（整数）,
  "scene_index": 场景索引（整数）,
  "source": "novel",
  "active_persona": "本场景中角色活跃的身份（对复合身份角色重要，如同一角色的不同时间线人格）",

  "dialogue_samples": ["原文台词1", "原文台词2"],
  "language_traits": "观察到的语言特征（口头禅/句式/语域/幽默方式）",

  "emotional_state": "当前情绪状态",
  "inferred_motivation": "推断的行为动机",
  "internal_conflict": "如有内心冲突，描述之；无则填空字符串",

  "relationships": [
    {"target": "角色B", "dynamic": "本场景中的关系表现", "shift": "相比之前有无变化，无则为 null"}
  ],

  "actions_taken": "做了什么",
  "actions_avoided": "选择不做什么（负面约束信号）",
  "decision_logic": "决策背后的逻辑推断",

  "arc_marker": "是否为角色发展的关键节点，如何改变了角色",

  "knowledge_scope": {
    "facts_revealed": ["本场景中角色获知的新信息"],
    "facts_hidden": ["本场景中角色不知道但读者知道的信息"]
  }
}
```

## 重要提醒

- `character` 字段必须使用任务中提供的 canonical_name（正式名），不要用场景文本中出现的别名
- `dialogue_samples` 必须保留原文，不要改写或总结
- `actions_avoided` 需要特别注意：角色在这种场景下有机会做 X 但选择了不做
- `active_persona`：对于有多重身份的角色（如同一人物的过去/未来化身），填写本场景中活跃的人格/身份名；普通角色直接填写 canonical_name
- 只提取实际出现在本场景中的角色；不出现则不输出对应 note
"""


def build_multi_scene_extraction_prompt(
    scene_characters: list[tuple[str, str]],
    chapter_index: int,
    scene_index: int,
    location: str,
    time: str,
    scene_content: str,
) -> str:
    """Build user prompt for multi-character scene extraction.

    Args:
        scene_characters: List of (canonical_name, alias_in_scene) for target characters
            that appear in this scene according to character_aliases.json.
        chapter_index: Chapter index (integer).
        scene_index: Scene index within the chapter (integer).
        location: Scene location from scenes_manifest.json.
        time: Scene time from scenes_manifest.json.
        scene_content: Raw scene text.

    Returns:
        Formatted prompt string.
    """
    char_lines = "\n".join(
        f'  - canonical_name: 「{canonical}」  （在本场景文本中出现为：「{alias}」）'
        for canonical, alias in scene_characters
    )
    return f"""## 任务

从以下小说场景中，提取所有列出的目标角色的信息。

## 目标角色（alias 为角色在本场景文本中的实际称呼）

{char_lines}

## 场景信息

- 章节索引 (chapter_index): {chapter_index}
- 场景索引 (scene_index): {scene_index}
- 地点: {location}
- 时间: {time}

## 场景内容

---开始---
{scene_content}
---结束---

## 输出格式

严格按照 SceneExtractionResult JSON 格式输出。每个实际出现的目标角色输出一条 note。未出现的角色不要输出。"""

SYSTEM_PROMPT = """你是一个专业的角色分析师。你的任务是从给定的小说章节中提取关于特定角色的结构化信息。

## 输出要求

以 JSON 格式输出，包含以下字段:

```json
{
  "character_scene_note": {
    "character": "角色名",
    "chapter_index": 章节索引,
    "scene_index": 场景索引,
    "source": "novel" | "film" | "artbook",

    "dialogue_samples": ["原文台词1", "原文台词2"],
    "language_traits": "观察到的语言特征 (口头禅/句式/语域/幽默方式)",

    "emotional_state": "当前情绪状态",
    "inferred_motivation": "推断的行为动机",
    "internal_conflict": "如有内心冲突, 描述之",

    "relationships": [
      {"target": "角色B", "dynamic": "本场景中的关系表现", "shift": "相比之前有无变化"}
    ],

    "actions_taken": "做了什么",
    "actions_avoided": "选择不做什么 (负面约束信号)",
    "decision_logic": "决策背后的逻辑推断",

    "arc_marker": "是否为角色发展的关键节点, 如何改变了角色",

    "knowledge_scope": {
      "facts_revealed": ["本场景中角色获知的新信息"],
      "facts_hidden": ["本场景中角色不知道但观众知道的信息"]
    }
  }
}
```

## 重要提醒

- `dialogue_samples` 必须保留原文，不要改写或总结
- `actions_avoided` 需要特别注意：角色在这种场景下有机会做X但选择了不做
- 如果该角色未在章节中出现，输出: "NOT_PRESENT"
- 使用中性、客观的语言描述
- 一个章节可能包含该角色的多个场景，分别提取"""


def build_extraction_prompt(character_name: str, chapter_index: int, chapter_title: str, chapter_content: str) -> str:
    """Build the user prompt for character extraction.

    Args:
        character_name: Name of the character to extract
        chapter_index: Index of the chapter
        chapter_title: Title of the chapter
        chapter_content: Full text content of the chapter

    Returns:
        Formatted prompt string
    """
    return f"""## 任务

从以下小说章节中提取关于「{character_name}」的角色信息。

## 章节信息

- 章节索引: {chapter_index}
- 章节标题: {chapter_title}

## 章节内容

---开始---
{chapter_content}
---结束---

## 输出格式

严格按照上述 JSON 格式输出。如果该角色未出现，输出 "NOT_PRESENT"。"""
