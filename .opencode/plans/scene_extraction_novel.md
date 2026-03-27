# Plan A: 小说场景提取 (Novel Scene Extraction)

> **目标**: 从小说文档提取结构化场景 | **输入**: material.yaml + 小说文件 | **输出**: scene_metadata.json + scene_XXX.txt

---

## 1. 架构概览

```
material.yaml + novel_file
    │
    ▼
┌──────────────────────────────────────────────┐
│  Layer 0: 文档解析 Tools (Python Classes)    │
│  ├─ TxtParser / EpubParser / MdParser        │
│  └─ list_chapters() / read_chapter()         │
│  输出: Chapter对象(内存) → 直接给Agent消费    │
└──────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────┐
│  Step 1: 场景切分 Agent (LLM-based)          │
│  输入: 章节文本                               │
│  输出: scene_boundaries.json (边界列表)       │
└──────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────┐
│  Step 2: 场景提取 Agent (LLM-based)          │
│  输入: 章节文本 + boundaries                  │
│  输出: scene_metadata.json + scene_XXX.txt   │
└──────────────────────────────────────────────┘
```

---

## 2. Layer 0: 文档解析 Tools

### 2.1 设计原则
- **非Agent**: 纯Python工具类，无LLM调用
- **无文件输出**: 直接返回内存对象，Agent即时消费
- **多格式支持**: txt/epub/md

### 2.2 接口定义

```python
from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import List

class ChapterInfo(BaseModel):
    index: int
    title: str
    length: int  # Unicode codepoint count, not byte count

class NovelDocumentParser(ABC):
    """小说文档解析器基类"""
    
    def __init__(self, file_path: str, encoding: str = "utf-8"):
        self.file_path = file_path
        self.encoding = encoding
    
    @abstractmethod
    def list_chapters(self) -> List[ChapterInfo]:
        """列出所有章节，不读取内容"""
        pass
    
    @abstractmethod
    def read_chapter(
        self,
        chapter_index: int,
        start_offset: int = 0,
        end_offset: int = None
    ) -> str:
        """
        读取指定章节文本

        Args:
            chapter_index: 章节索引
            start_offset: 起始位置（Unicode codepoint偏移，从0开始）
            end_offset: 结束位置（Unicode codepoint偏移，None表示读到末尾）

        Returns:
            指定范围的章节文本
        """
        pass
    


# 具体实现（在使用处根据文件扩展名直接实例化）
class TxtParser(NovelDocumentParser): ...
class EpubParser(NovelDocumentParser): ...
class MarkdownParser(NovelDocumentParser): ...

# 使用示例:
# if file_path.endswith('.txt'): parser = TxtParser(file_path)
# elif file_path.endswith('.epub'): parser = EpubParser(file_path)
```

### 2.3 实现细节

**缓存策略**:
- **元数据缓存**: 缓存章节byte offset位置，不缓存内容
- **缓存失效**: 通过文件mtime + size检测变化，自动重建缓存
- **懒加载**: 首次调用`list_chapters()`时构建缓存，后续直接使用

**TxtParser**:
- 章节识别: 通过**自定义正则表达式**（用户需提供，不内置默认模式）
- 不同小说章节格式差异大，内置模式容易失效
- 返回byte offset位置，支持快速seek随机读取

**EpubParser**:
- 使用 ebooklib 或类似库
- 提取 OPF 元数据
- 按 HTML 文件切分章节，记录文件索引

**MarkdownParser**:
- 基于 heading (# ## ###) 切分
- 支持 YAML frontmatter 元数据
- 记录行号offset

---

## 3. Step 1: 场景切分 Agent

### 3.1 目标
识别章节内的场景边界，不生成内容，只输出边界列表。

### 3.2 输入
```json
{
  "chapter_index": 0,
  "chapter_title": "第一章 降临",
  "chapter_text": "[完整章节文本]",
  "target_tokens_per_scene": 3000
}
```

### 3.3 输出 (文本匹配后的最终边界)

Agent输出文本片段，工具端匹配后生成最终边界：

```json
{
  "chapter_index": 0,
  "chapter_title": "第一章 降临",
  "total_chars": 15234,
  "scenes": [
    {
      "scene_id": "c01_s01",
      "start_char": 0,
      "end_char": 2156,
      "boundary_type": "location_change",
      "boundary_reason": "从月读空间切换到地球某城市",
      "estimated_tokens": 3234,
      // 以下为匹配信息（调试用）
      "matched_before": "...与此同时，在月读空间里。",
      "matched_after": "辉夜站在街头，困惑地看着四周..."
    },
    {
      "scene_id": "c01_s02",
      "start_char": 2156,
      "end_char": 5234,
      "boundary_type": "time_jump",
      "boundary_reason": "三天后",
      "estimated_tokens": 4617
    }
  ]
}
```

**流程**: Agent输出文本片段 → 工具端文本匹配 → 生成精确字符位置

### 3.4 切分信号 (按优先级)

1. **地点变化**: "与此同时，在月读空间..." / "她回到了自己的房间..."
2. **时间跳跃**: "三天后..." / "第二天早上..." / "一个月后..."
3. **POV切换**: 视角人物改变 (第一人称/第三人称有限)
4. **叙事断裂**: 空行、分隔符、章节子标题等

### 3.5 长度控制策略

- **目标**: 2k-5k tokens/场景（放宽上限以保留叙事连贯性）
- **范围**: 1.5k-6k
  - **低于1.5k**: 与相邻场景合并
  - **超过6k**: 在对话/段落间隙二次切分
- **理由**: 
  - 避免过度切分破坏叙事节奏
  - 现代LLM上下文充足（200k+），5-6k仅占2.5-3%
  - 小说场景自然长度差异大，应允许适度弹性

### 3.6 非叙事段落处理

**识别方式**: 由**场景切分 Agent 自动识别**（无需人工预先标记）

**识别标准**:
- 警告/注意事项（caution page）
- 目录（TOC）
- 版权页/献词（colophon）
- 后记/作者注（afterword - 需区分是否包含叙事内容）
- 其他非故事内容（人物介绍表、时间线说明等）

**处理方式**:
```json
{
  "scene_id": "c00_s01",
  "start_char": 0,
  "end_char": 1250,
  "boundary_type": "non_narrative",
  "boundary_reason": "警告页，不含叙事内容",
  "estimated_tokens": 1875
}
```

**场景提取时的处理**:
- 整段作为一个场景输出
- `scene_info.location`: `"N/A"`
- `scene_info.time_of_day`: `"N/A"`
- `characters.present`: `[]` (空列表)
- `characters.mentioned`: `[]` (空列表)
- 场景文本正常保存，供后续流程参考
- 后续角色提取 Pipeline 会跳过 `boundary_type: "non_narrative"` 的场景

**为什么不让代码规则检测**:
- AGENT.md: "NEVER implement code-based automatic detection"
- 不同作品格式差异大，规则容易误判（如 "あとがき" 可能包含角色信息）
- LLM 通过上下文理解判断更准确

### 3.7 渐进式小批量切分

**问题**: 一次给模型太多文本会导致输出质量下降（场景边界判断不准确）

**解决方案**: 限制每次处理的文本量，让模型一次只生成1-3个场景切分

```python
async def segment_chapter(chapter_text: str) -> List[SceneBoundary]:
    """
    渐进式切分章节，小批量处理确保质量。
    """
    processed_boundaries = []  # 已确认的场景边界
    remaining_text = chapter_text  # 待处理的剩余文本
    current_offset = 0  # 当前在原文中的偏移
    previous_summary = ""  # 已处理内容的摘要
    
    while remaining_text:
        # 准备输入：已处理部分(摘要) + 待处理部分(限制长度)
        context_window = build_context_window(
            processed_summary=previous_summary,
            remaining_text=remaining_text,
            max_new_text=8000,  # 约5k-6k tokens，给模型足够上下文但不超载
            max_scenes_hint=3   # 建议生成1-3个场景
        )
        
        # 调用场景切分Agent
        result = await scene_segmentation_agent.run(
            context=context_window,
            current_offset=current_offset
        )
        
        # 处理返回的场景边界
        for boundary in result.new_boundaries:
            # 文本匹配找到精确位置
            start_char, end_char = find_boundary_by_text_match(
                chapter_text,
                boundary.before_text,
                boundary.after_text,
                search_start=current_offset,
                min_length=boundary.min_length,
                max_length=boundary.max_length
            )
            
            processed_boundaries.append({
                "scene_id": f"c{chapter_index:02d}_s{len(processed_boundaries)+1:02d}",
                "start_char": start_char,
                "end_char": end_char,
                "boundary_type": boundary.boundary_type
            })
        
        # 更新状态
        if result.has_more:
            # 还有剩余内容，继续处理
            # next_position 告诉工具下次从哪里继续
            processed_length = result.next_position - current_offset
            current_offset = result.next_position
            remaining_text = chapter_text[current_offset:]
            previous_summary = update_summary(previous_summary, context_window[:processed_length])
        else:
            # 全部处理完成
            break
    
    return processed_boundaries

def build_context_window(
    processed_summary: str,
    remaining_text: str,
    max_new_text: int,
    max_scenes_hint: int
) -> str:
    """
    构建模型输入上下文：
    - 已处理内容的简要摘要（保持连贯性）
    - 待处理的新文本（限制长度，避免超载）
    """
    new_text = remaining_text[:max_new_text]
    
    return f"""
[已处理内容摘要]
{processed_summary}

[待处理新文本 - 建议切分为{max_scenes_hint}个以内场景]
{new_text}

[指令]
请在上述新文本中识别1-{max_scenes_hint}个场景边界。
如果还有更多内容未处理完，请标注has_more=true并给出next_position。
"""
```

**关键设计**:
- **小批量处理**: 每次只给模型~5k-6k tokens新文本，避免超载
- **分离已处理/待处理**: 已处理部分用摘要代替，待处理部分用原文
- **限制场景数量**: 每次要求模型只生成1-3个场景，保证质量
- **渐进推进**: 处理完一批后，更新摘要并继续下一批
- **文本匹配**: 模型输出文本片段，工具端匹配找到精确位置
- **连贯性保持**: 通过摘要让模型了解前文脉络

### 3.8 Prompt模板

```
你是一个专业的叙事分析助手。请分析以下小说内容，识别1-3个场景边界。

**输入结构**:
[已处理内容摘要]
{processed_summary}

[待处理新文本 - 约5k-6k tokens]
{new_text_chunk}

场景定义：在一个相对连续的时间和空间中发生的叙事单元。

切分信号（按优先级排序）：
1. 地点变化
2. 时间跳跃（超过1小时）
3. 视角人物(POV)切换
4. 叙事断裂标记

**任务要求**：
1. 在新文本中识别**1-3个**场景边界（不要一次生成太多，保证质量）
2. 为每个场景标注切分类型和原因
3. **定位方式**：输出before_text（场景结束前30-50字符）+ after_text（场景开始后30-50字符）
4. **歧义避免**：确保before_text+after_text的组合在新文本中唯一确定切分点
5. 估算场景长度范围(min_length, max_length)，用于验证
6. 判断：是否还有未处理完的内容？输出has_more和next_position

**特殊处理 - 非叙事段落**:
如果新文本主要是非叙事内容（如警告页、目录等）：
- 输出一个场景，boundary_type = "non_narrative"

**重要**：不要输出数字位置，模型不擅长精确计数！

返回格式 (JSON):
{
  "new_boundaries": [
    {
      "before_text": "...与此同时，在月读空间里。",  // 30-50字符
      "after_text": "辉夜站在街头，困惑地看着四周...",  // 30-50字符
      "min_length": 1800,
      "max_length": 2200,
      "boundary_type": "location_change",
      "boundary_reason": "从月读空间切换到地球某城市",
      "estimated_tokens": 3000
    }
  ],
  "has_more": true,        // 是否还有未处理内容
  "next_position": 4500,   // 下次应该从原文的哪个位置继续（字符数）
  "summary_update": "..."  // 本次处理内容的简要摘要，追加到processed_summary
}

**工具端处理**：
- 在原文中搜索before_text+after_text匹配
- 用min/max_length做sanity check
- 根据next_position准备下一批文本
```

---

## 4. Step 2: 场景提取 Agent

### 4.1 目标
基于边界提取场景文本，生成完整元数据。

### 4.2 输入
```json
{
  "chapter_index": 0,
  "chapter_title": "第一章 降临",
  "chapter_text": "[完整章节文本]",
  "boundaries": [...],
  "material_info": {
    "project_name": "超时空辉夜姬",
    "source_file": "超时空辉夜姬.txt"
  }
}
```

### 4.3 输出

**文件1: scene_c01_s01.txt**
```
# scene_c01_s01.txt
# Source: 超时空辉夜姬.txt
# Chapter: 第一章 降临 (Index: 0)
# Scene ID: c01_s01
# Location: 地球-某城市-街头
# Characters: 辉夜
# Tokens: ~1867
# ---

[场景完整文本，保留原始格式]
```

**文件2: scene_metadata.json (累积写入)**
```json
{
  "metadata": {
    "version": "1.0",
    "generated_at": "2026-03-26T10:00:00Z",
    "project": "超时空辉夜姬",
    "source_file": "超时空辉夜姬.txt",
    "parser_type": "txt",
    "total_scenes": 156,
    "total_chapters": 12
  },
  "scenes": [
    {
      "scene_id": "c01_s01",
      "source": {
        "type": "novel",
        "file": "超时空辉夜姬.txt",
        "chapter_index": 0,
        "chapter_title": "第一章 降临"
      },
      "content": {
        "file_path": "scene_c01_s01.txt",
        "char_count": 1245,
        "token_estimate": 1867,
        "hash": "sha256:abc123..."
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
      },
      "segmentation": {
        "boundary_type": "location_change",
        "start_char": 0,
        "end_char": 1245
      }
    }
  ]
}
```

### 4.4 Prompt模板

```
你是一个专业的场景信息提取助手。给定章节文本和场景边界，请提取场景的完整信息和元数据。

任务：
1. 按边界提取场景的完整文本内容
2. 分析并填写元数据字段
3. 识别场景中出现的角色（使用标准角色名）

元数据字段：
- location: 场景发生的地点（尽可能具体，如"地球-东京-新宿街头"）
- time_of_day: 一天中的时间（早晨/中午/傍晚/夜晚/深夜/未知）
- timeline_position: 在故事时间线中的位置（如"arc_1_chapter_1"）
- characters.present: 实际在场、有行动或台词的角色（使用标准名）
- characters.mentioned: 被提及但未在场的角色
- characters.pov_character: 视角人物（如果是第三人称有限视角）

注意：
- 保持文本原始格式，包括换行和段落
- 角色名按material.yaml中的标准名填写
- 如果无法确定某个字段，使用"unknown"

章节文本:
{chapter_text}

当前场景边界:
{current_boundary}
```

### 4.5 Token估算策略

**方法**: 字符数 × 1.5
- 中文字符：每个约1.5-2个token
- 简单估算，无需引入tiktoken

**用途**:
- 场景长度控制（目标2k-4k tokens）
- 预估处理成本
- 检测过长场景需要二次切分

---

## 5. 数据格式规范

### 5.1 输出目录结构

```
output/
└── novel/
    ├── scene_metadata.json       # 所有场景元数据索引
    ├── scene_c01_s01.txt        # 场景文本文件
    ├── scene_c01_s02.txt
    ├── scene_c02_s01.txt
    └── ...
```

### 5.2 Scene ID格式

```
c{章节索引:02d}_s{场景索引:02d}

示例:
- c01_s01 = 第1章第1场景
- c12_s05 = 第12章第5场景
```

### 5.3 scene_metadata.json Schema

```python
class SceneContent(BaseModel):
    file_path: str
    char_count: int
    token_estimate: int
    hash: str  # SHA256

class SceneInfo(BaseModel):
    location: str
    time_of_day: str
    timeline_position: str

class Characters(BaseModel):
    present: List[str]
    mentioned: List[str]
    pov_character: Optional[str]

class Segmentation(BaseModel):
    boundary_type: str  # location_change/time_jump/pov_switch/narrative_break
    start_char: int
    end_char: int

class Scene(BaseModel):
    scene_id: str
    source: dict
    content: SceneContent
    scene_info: SceneInfo
    characters: Characters
    segmentation: Segmentation

class SceneMetadata(BaseModel):
    metadata: dict
    scenes: List[Scene]
```

---

## 6. 实现计划

### 6.1 模块结构

```
yorishiro/
├── parsers/                        # Layer 0: 文档解析Tools
│   ├── __init__.py
│   ├── base.py                     # NovelDocumentParser基类
│   ├── txt_parser.py
│   ├── epub_parser.py
│   └── markdown_parser.py
├── agents/
│   └── novel/
│       ├── __init__.py
│       ├── scene_segmentation.py   # Step 1: 场景切分Agent
│       └── scene_extraction.py     # Step 2: 场景提取Agent
├── pipelines/
│   └── novel_scene_extraction.py   # Pipeline主控
├── models/
│   └── scene_models.py             # Pydantic模型
└── utils/
    └── hash.py                     # 文本哈希工具
```

### 6.2 框架选择

**PydanticAI + LiteLLM**

- **PydanticAI**: 类型安全的Agent框架，强制结构化输出验证
- **LiteLLM**: 统一多供应商API接口(OpenAI/Claude/本地模型)
- **强制工具输出**: 使用 `result_type` 指定 Pydantic 模型，LLM 必须通过 tool calling 返回结构化数据

```python
# 示例Agent定义 - 强制使用工具输出
from pydantic_ai import Agent
from pydantic import BaseModel, Field

class SceneBoundary(BaseModel):
    """单个场景边界（模型输出）"""
    before_text: str = Field(description="场景结束前30-50字符，用于文本匹配定位")
    after_text: str = Field(description="场景开始后30-50字符，用于文本匹配定位")
    min_length: int = Field(description="场景最小可能长度（字符数），用于验证匹配结果")
    max_length: int = Field(description="场景最大可能长度（字符数），用于验证匹配结果")
    boundary_type: str = Field(description="切分类型: location_change/time_jump/pov_switch/narrative_break/non_narrative")
    boundary_reason: str = Field(description="切分原因说明")
    estimated_tokens: int = Field(description="估算token数（字符数×1.5）")

class SegmentationBatchResult(BaseModel):
    """小批量切分结果（单次调用返回）"""
    new_boundaries: list[SceneBoundary] = Field(description="本次识别的新场景边界（1-3个）")
    has_more: bool = Field(description="是否还有未处理的后续内容")
    next_position: int = Field(description="下次应该从原文的哪个字符位置继续处理")
    summary_update: str = Field(description="本次处理内容的简要摘要，用于保持上下文连贯性")

scene_segment_agent = Agent(
    model="claude-3-5-sonnet",
    result_type=SegmentationResult,  # 强制结构化输出
    system_prompt="你是一个专业的叙事分析助手..."
)

# 使用方式 - PydanticAI 会自动处理 tool calling
result = await scene_segment_agent.run(
    user_prompt=f"分析以下章节片段...",
    deps={"start_offset": 0, "previous_context": ""}
)
# result.data 已经是验证后的 SegmentationResult 对象
```

**优势**:
- **类型安全**: Pydantic 自动验证输出格式
- **质量保障**: LLM 必须通过 tool calling 返回，避免自由文本中的格式错误
- **字段说明**: 通过 `Field(description=...)` 提供每个字段的详细说明，提升 LLM 理解
- **无需手动解析**: 不需要手动提取 JSON，直接使用类型化的 Pydantic 对象

### 6.3 开发任务分解

| 任务 | 描述 | 优先级 | 预估工时 |
|------|------|--------|----------|
| 1 | 集成PydanticAI + LiteLLM | 高 | 3h |
| 2 | 设计Document Parser基类接口 | 高 | 2h |
| 3 | 实现TxtParser(含缓存) | 高 | 4h |
| 4 | 实现EpubParser | 中 | 4h |
| 5 | 实现MarkdownParser | 低 | 2h |
| 6 | 设计场景切分Agent(PydanticAI) | 高 | 5h |
| 7 | 设计场景提取Agent(PydanticAI) | 高 | 4h |
| 8 | 实现大章节流式处理 | 高 | 4h |
| 9 | 设计数据模型(Pydantic) | 高 | 2h |
| 10 | 实现Pipeline主控(含--force) | 高 | 4h |
| 11 | 编写测试用例 | 中 | 4h |
| 12 | 集成测试 | 高 | 3h |

**总计**: ~41工时

---

## 7. 与Plan B的接口

Plan A的输出是Plan B的输入：

```
Plan A (本Plan)
    │
    ├── scene_metadata.json  ──────┐
    └── scene_XXX.txt             │
                                  ▼
                          Plan B (Character Extraction)
```

Plan B期望的Scene格式详见 `character_extraction_plan.md` §2.1。

---

## 8. 已确认设计决策

### 8.1 章节缓存策略 ✓
- **不缓存内容**，只缓存章节byte offset位置
- **缓存失效**: 通过文件mtime + size检测变化
- **懒加载**: 首次调用`list_chapters()`时构建
- **按格式优化**: TXT用byte offset快速seek，EPUB用内部索引

### 8.2 小批量渐进式切分策略 ✓
- **质量优先**: 限制每次输入文本量（~5k-6k tokens），避免模型超载
- **小批量输出**: 每次要求模型只生成1-3个场景边界，保证切分质量
- **分离已处理/待处理**: 
  - 已处理部分：用摘要(summary)代替原文，保持连贯性
  - 待处理部分：限制长度的新文本(~5k-6k tokens)
- **渐进推进**: 处理完一批后，更新摘要，继续下一批
- **统一循环**: 所有章节（无论长短）使用相同逻辑，无需特殊处理

### 8.3 文本匹配定位策略 ✓
- **Agent输出**: `before_text` + `after_text`（前后文本片段），不输出数字位置
- **工具端匹配**: 在原文中搜索匹配，找到精确切分点
- **歧义避免**: 模型确保切分点在已读文本中无歧义（不会出现两个匹配）
- **长度验证**: `min_length`/`max_length` 用于 sanity check 验证匹配结果合理性
- **允许差异**: 匹配时允许中间有空白字符差异

### 8.3 Token估算策略 ✓
- **方法**: 字符数 × 1.5（简单估算，无需tiktoken）
- **用途**: 场景长度控制、成本预估、检测过长场景

### 8.4 输出覆盖策略 ✓
- **默认增量**: 检查文件hash，只处理变化的章节
- **--force标志**: 强制全量重新生成
- **适用场景**: 调试时全量重跑，日常运行增量更新

---

**下一步**: 开始实现Layer 0文档解析Tools。
