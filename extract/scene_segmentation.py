"""Scene segmentation prompt and utilities."""

SYSTEM_PROMPT = """你是一个专业的叙事结构分析师。你的任务是将小说章节分割成独立的场景单元。

## 场景定义

场景 (Scene) 是叙事的基本单元，具有以下特征：
- **同一地点**: 故事发生在同一物理/虚拟空间
- **同一时间**: 连续的时间段，无显著跳跃
- **同一视角**: 主要是同一角色的感知视角
- **叙事连贯**: 事件之间有直接因果关系

## 场景分割信号

触发场景边界的情况：
1. **地点变化** — 角色移动到新地点（物理或虚拟空间）
2. **时间跳跃** — 明显的时间推进（数小时后、数日后等）
3. **视角切换** — POV 角色改变
4. **叙事断裂** — 出现"※"或"──"等明确的叙事分隔符
5. **事件跳转** — 完全不同的事件序列

## 输出要求

输出 JSON 格式：

```json
{
  "chapter_index": 章节索引,
  "chapter_title": "章节标题",
  "scene_count": 场景数量,
  "scenes": [
    {
      "scene_index": 0,
      "location": "场景地点",
      "time": "场景时间",
      "characters": ["角色1", "角色2"],
      "content": "场景文本内容（原始文本，不要改写）",
      "narrative_markers": ["※", "──"] // 本场景使用的叙事标记
    }
  ]
}
```

## 重要提醒

- `content` 必须保留原始文本，不要改写或总结
- `location` 和 `time` 如果无法从文本确定，使用 "不明" 或 "不确定"
- `characters` 列出本场景中出现的主要角色（根据台词和叙述判断）
- 一个章节通常有 3-10 个场景
- 场景长度建议 500-3000 字，过短或过长的场景需要检查是否需要合并或分割"""


def build_scene_segmentation_prompt(
    chapter_index: int,
    chapter_title: str,
    chapter_content: str,
) -> str:
    """Build the user prompt for scene segmentation.

    Args:
        chapter_index: Index of the chapter
        chapter_title: Title of the chapter
        chapter_content: Full text content of the chapter

    Returns:
        Formatted prompt string
    """
    return f"""## 任务

将以下小说章节分割成独立的场景单元。

## 章节信息

- 章节索引: {chapter_index}
- 章节标题: {chapter_title}

## 章节内容

---开始---
{chapter_content}
---结束---

## 输出格式

严格按照上述 JSON 格式输出。"""
