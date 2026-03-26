"""Character extraction prompt templates for LLM extraction."""

SYSTEM_PROMPT = """你是一个专业的角色分析师。你的任务是从给定的小说章节中提取关于特定角色的结构化信息。

## 输出要求

以 YAML 格式输出，包含以下字段:

```yaml
character_scene_note:
  character: "角色名"
  chapter_index: 章节索引
  source: "novel" | "film" | "artbook"

  # 台词与语言风格
  dialogue_samples: ["原文台词1", "原文台词2"]   # 保留原文, 不要总结
  language_traits: "观察到的语言特征 (口头禅/句式/语域/幽默方式)"

  # 情绪与动机
  emotional_state: "当前情绪状态"
  inferred_motivation: "推断的行为动机"
  internal_conflict: "如有内心冲突, 描述之"

  # 关系动态
  relationships:
    - target: "角色B"
      dynamic: "本场景中的关系表现"
      shift: "相比之前有无变化"

  # 行为模式
  actions_taken: "做了什么"
  actions_avoided: "选择不做什么 (负面约束信号)"
  decision_logic: "决策背后的逻辑推断"

  # Arc 信号
  arc_marker: "是否为角色发展的关键节点, 如何改变了角色"

  # 知识边界 (运行时 RAG 使用)
  knowledge_scope:
    facts_revealed: ["本场景中角色获知的新信息"]
    facts_hidden: ["本场景中角色不知道但观众知道的信息"]
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

严格按照上述 YAML 格式输出。如果该角色未出现，输出 "NOT_PRESENT"。"""
