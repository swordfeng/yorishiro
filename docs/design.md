# Yorishiro（依り代）— 技术设计

> **版本**: 0.3 · **状态**: 草案 · **最后更新**: 2026-03-31

产品目标与规格见 [requirements.md](./requirements.md)。

---

## 1. Phase 1: 提取层 (Extraction)

### 1.1 影片处理链路

**核心概念区分**: PySceneDetect 检测的是**分镜** (shot, 镜头切换单元), 而非叙事**场景** (scene)。一场对话可能包含数十个正反打分镜。需要单独的 LLM 步骤将分镜合并为叙事场景。

```
影片文件
  │
  ├─ Layer 0A: 视频处理 (非LLM, 纯Python)
  │   ├─ 分镜检测 ─────── PySceneDetect detect-adaptive → 分镜列表 + 时间戳
  │   └─ 关键帧提取 ───── CLIP embedding 选帧
  │        ├─ Step 1 用: 每分镜 1 帧 (代表帧, 控制 token)
  │        └─ Step 2 用: 每场景 3-5 帧 (余弦距离选最大语义变化帧)
  │
  ├─ Layer 0B: 音频分析 (非LLM / 轻量本地模型)
  │   ├─ SpeechPipeline
  │   │    ├─ VAD ──────── Silero-VAD, 切出语音段, 过滤静音/音乐段
  │   │    ├─ 说话人分离 ─ pyannote-audio 3.x → 局部 SP_A/SP_B
  │   │    ├─ Speaker Bank 提取 d-vector, 全局聚类 → 一致 SPKR_XXX
  │   │    ├─ STT ─────── per-segment 转录 (比整流更准确, 避免重叠干扰)
  │   │    └─ 情绪/语调 ── per-句话: 情绪类别 + pitch + 语速 + 音量
  │   ├─ SoundEventDetector
  │   │    └─ CLAP 零样本检测: 非语音人声 + 环境音 + 音效
  │   └─ MusicAnalyzer
  │        ├─ Demucs ───── 人声/伴奏分离
  │        ├─ 音乐段检测 ─ BGM vs 插入曲 (有无歌词)
  │        ├─ 歌词提取 ─── Whisper 对人声轨道单独处理
  │        └─ 特征分析 ─── BPM / 调性 / 乐器 / valence / arousal
  │
  ├─ Step 1: 分镜合并 Agent (LLM, 小批量渐进式)
  │   输入: 每批 10-15 分镜 + 每镜 1 帧 + 每镜音频摘要
  │   判断依据: 视觉连续性 / 音频连续性 / 叙事逻辑 (正反打等剪辑结构)
  │   输出: scene_groupings (哪些分镜属于同一叙事场景)
  │
  └─ Step 2: VLM 场景分析 Agent
       输入: 合并后场景 + 3-5 帧关键帧 + 完整音频分析
       输出: scene_metadata.json (source_type: "film") + scene_fs_XXX.txt
```

**所有模型均可配置**, 支持本地和云端切换:

| 环节 | 本地默认 | 云端备选 | 备注 |
|------|---------|---------|------|
| 分镜检测 | PySceneDetect detect-adaptive | — | 阈值可调 |
| 关键帧 embedding | CLIP ViT-L/14 | — | 或 ViT-B/32 |
| VAD | Silero-VAD | — | |
| 说话人分离 | pyannote/speaker-diarization-3.1 | — | 需 HF token |
| Speaker Embedding | pyannote SpeakerEmbedding | Resemblyzer | 用于跨场景声纹 |
| STT | faster-whisper large-v3 | Deepgram Nova-3 | 云端对背景音更鲁棒 |
| 情绪分析 | emotion2vec_plus_large | — | 中日文效果好 |
| 声音事件检测 | CLAP larger_clap_general | — | 零样本 |
| 音乐分离 | Demucs htdemucs | — | |
| 音乐特征 | Essentia | — | BPM/调性/乐器 |
| 分镜合并 LLM | claude-sonnet-4-6 | gemini-2.0-flash / gpt-4o | |
| 场景分析 VLM | claude-opus-4-6 | gemini-2.0-flash / gpt-4o | 多模态 |

**Layer 0B 音频分析详解**:

*SpeechPipeline — 跨场景声纹联系*:
- pyannote 输出局部说话人 ID (每场景独立), 无法直接跨场景关联
- 为每个说话人段提取 d-vector, 维护全局 Speaker Bank
- 余弦相似度匹配: 高于阈值 (默认 0.75, 可配置) → 合并为已有 SPKR_XXX; 否则新建
- Step 2 VLM 场景分析建立 `SPKR_XXX → 角色名` 映射; 已确认映射在后续场景作为 prior

*SoundEventDetector — 检测范围*:
- 非语音人声: 哭泣、笑声、叹气、喘气、呼吸、心跳、呻吟
- 环境音: 机械音、电子音、人群、自然音 (雨/风/水)
- 音效: 武器、爆炸、玻璃破碎、门开关

*MusicAnalyzer — 为什么重要*:
- 日系作品插入曲往往标志角色关键情感节点
- 影片的情绪氛围信号大量承载于音乐而非文字, 对角色提取具有重要参考价值

**Step 1 分镜合并 — 小批量渐进式**:

与小说场景切分 Agent 对应。分镜已有 ID, 无需文本匹配定位:
- 每批传入 10-15 个分镜 (每镜 1 帧 + 音频摘要) 和已处理内容摘要
- Agent 判断分组, `is_complete: false` 表示批次边界截断, 下批以 `partial_group` 续接
- 已处理部分用摘要代替, 保持叙事上下文连贯性

**Step 2 输出格式 (`scene_fs_XXX.txt`)**:

```
# scene_fs003.txt
# Source: movie.mkv
# Scene ID: fs003
# Timestamp: 00:05:32.100 - 00:07:45.800
# Location: 地球-东京-室内某处
# Characters: 辉夜, 彩叶
# ---

[视觉描述]
昏暗的室内空间。辉夜背对窗户站立, 逆光使表情难以辨认。彩叶坐在沙发上, 双手交握, 目光直视辉夜。

[音乐]
00:05:32-00:07:45 | BGM | 钢琴独奏, BPM 58, 情绪: 压抑与等待, valence 低, arousal 低

[对话与声音]
00:05:38 彩叶 [平静, 音量低]: "你早就知道了, 对吧。"
00:05:44 [非语音: 辉夜短促吸气]
00:05:47 辉夜 [克制, 声线微颤]: "……知道又怎样。"
00:05:52 [环境音: 窗外远处车声, 低频持续]
00:06:03 彩叶 [情绪: 悲伤转平静]: "我只是想让你亲口说出来。"
00:06:11 [非语音: 长时间沉默, 约8秒]
00:06:19 辉夜 [极低声, 近耳语]: "……对不起。"
```

格式规则: 时间戳精确到毫秒; `[非语音]`/`[环境音]` 与对话行按时序混排; 音乐段全局描述置于对话区块之前, 场景内切换时在对话行中插入标注行。

Scene ID 格式: `fs{全局索引:03d}` (影片无章节结构, 使用平铺索引)。

**缓存策略**: Layer 0 全部输出持久化, 缓存 key 为 `sha256(mtime + filesize)`:

```
processed/film/{video_hash}/
    ├── shots.json              # 分镜边界
    ├── frames/                 # 关键帧 JPEG (Step 1 代表帧 + Step 2 选取帧)
    ├── clip_embeddings.npz     # 所有帧的 CLIP embedding (按 shot_id 索引)
    ├── frame_index.json        # shot_id → 帧路径 (轻量索引, 不含 embedding)
    ├── transcript.json         # STT + 分离 + 情绪 (含全局 SPKR_XXX)
    ├── speaker_embeddings.npz  # 全局 Speaker Bank
    ├── speaker_map.json        # SPKR_XXX → 角色名 (Step 2 逐步建立)
    ├── sound_events.json       # 声音事件检测结果
    └── music_analysis.json     # BGM/插入曲分析结果
```

各模块缓存独立可单独跳过; `--force` 强制全量重跑 (与小说 pipeline 一致)。

### 1.2 小说处理链路

```
小说文本
  ├─ 场景分段 ──────────── LLM 辅助判断场景边界
  │    └─ 信号: 地点变化 / 时间跳跃 / POV 切换 / 章节边界
  ├─ 分段长度控制 ──────── 2k-4k tokens/段
  └─ 文本总结 + 角色提取 ─ 语言模型
       ├─ 场景总结
       └─ 角色级提取 (见 §1.4)
```

**场景分段策略**:
- 第一遍: 按章节自然分割
- 第二遍: 章节内如果超过 4k tokens, 用 LLM 判断场景边界做二次分割
- Prompt 要点: "识别地点变化、时间跳跃超过1小时、视角切换、或明显的叙事断裂"
- 输出: 每段标注 `scene_id`, `characters_present`, `location`, `timeline_position`

### 1.3 设定集 / 其他资料处理

```
设定集 (PDF/图册)
  ├─ 文字提取 ──────────── marker / MinerU
  ├─ 图片提取 ──────────── 同上 (自动分离)
  ├─ 结构化整理 ─────────── 按条目(角色/世界观/术语)归类
  └─ 入库 ─────────────── 直接存入索引层

其他资料 (创作者访谈/官方Q&A/...)
  └─ 文本提取后按来源标注, 直接入库
```

### 1.4 角色级提取 (Character-Centric Extraction)

> **关键设计**: 场景总结是事件中心的, 但 SOUL.md 需要角色中心的信息。  
> 必须在场景总结之外, 单独做一轮 per-character extraction。

**每个目标角色 × 每个场景, 提取以下维度**:

```yaml
character_scene_note:
  character: "角色名"
  scene_id: "s_042"
  source: "film" | "novel" | "artbook" | ...
  active_persona: "辉夜"              # 复合身份角色: 本场景活跃的身份 (见 §4.4)

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

  # 知识边界 (运行时 RAG 使用, 见 §2.4)
  knowledge_scope:
    facts_revealed: ["本场景中角色获知的新信息"]
    facts_hidden: ["本场景中角色不知道但观众知道的信息"]
```

**Prompt 设计要点**:
- 明确要求模型输出结构化 YAML/JSON
- 强调 `dialogue_samples` 必须保留原文, 这是语言风格分析的关键素材
- `actions_avoided` 需要特别 prompt: "角色在这种场景下有机会做X但选择了不做, 列举这些"
- 如果角色不在该场景中, 跳过 (不要生成空记录)

---

## 2. Phase 2: 索引层 (Memory / Index)

### 2.1 存储架构

```
┌────────────────────────────────────────────┐
│             向量数据库 (Qdrant)              │
│                                            │
│  Collection: scene_summaries               │
│    - vector: text-embedding-3-large        │
│    - payload: {scene_id, source_type,      │
│       characters, location, timeline_pos,  │
│       raw_segment_path}                    │
│                                            │
│  Collection: character_notes               │
│    - vector: text-embedding-3-large        │
│    - payload: {character, scene_id,        │
│       source_type, arc_marker, ...}        │
│                                            │
│  Collection: visual_frames (可选)           │
│    - vector: CLIP ViT-L/14                 │
│    - payload: {scene_id, frame_index,      │
│       timestamp, frame_path}               │
│                                            │
│  Collection: reference_entries             │
│    - 设定集/访谈等零散条目                    │
│    - vector + payload                      │
└────────────────────────────────────────────┘

原始文件存储: 本地文件系统 / S3
  - /raw/film/scenes/{scene_id}/frames/*.jpg
  - /raw/film/scenes/{scene_id}/transcript.txt
  - /raw/novel/segments/{scene_id}.txt
  - /raw/artbook/entries/{entry_id}/...
```

### 2.2 Embedding 选择

| 用途 | 模型 | 备注 |
|------|------|------|
| 文本语义检索 | `text-embedding-3-large` (OpenAI) | 中英双语效果好 |
| 文本语义检索 (开源) | `BGE-M3` (BAAI) | 支持多语言, 可本地部署 |
| 图片语义检索 | `CLIP ViT-L/14` | 用于关键帧检索 |

### 2.3 检索接口设计

```python
# 构建时 + 运行时 共用的检索接口
class MemorySearch:
    def search_by_text(self, query: str, filters: dict = None, top_k: int = 10):
        """语义搜索 + 结构化过滤
        
        filters 示例:
          {"character": "角色A", "source_type": "novel"}
          {"characters": {"$contains": ["角色A", "角色B"]}}  # 共同出场
          {"arc_marker": True}  # 只返回 arc 关键节点
          {"persona": "八千代"}  # 按人格身份过滤 (见 §4.4)
          {"knowledge_accessible_by": "辉夜"}  # 知识边界过滤 (见 §2.4)
        """

    def get_raw_segment(self, scene_id: str, source_type: str):
        """获取完整原始提取内容 (未经总结的原文/帧/字幕)"""

    def get_all_character_notes(self, character: str, sort_by: str = "timeline"):
        """获取某角色的所有场景级笔记, 按时间线排序"""

    def get_character_dialogues(self, character: str):
        """获取某角色的所有原文台词样本"""
```

### 2.4 运行时 RAG 专用设计

> **核心原则**: 构建时和运行时共享同一个索引库, 但运行时检索需要额外的约束层。

**为什么需要运行时 RAG (而不是只靠 SOUL.md)**:
- SOUL.md 是压缩后的人格摘要, 放入 system prompt, 控制角色的"是谁"
- 但对话中用户可能问到 SOUL.md 未覆盖的具体细节: 某个场景的具体经过、某段对话的原文、世界观设定的细节等
- 记忆系统在运行时作为 Agent 的"可查阅的记忆", 按需提供这些细节

**运行时检索架构**:

```
用户消息: "你还记得在月读里第一次和彩叶一起直播的时候吗?"
                │
                ▼
┌─────────────────────────────────────┐
│ 1. 查询构建 (Query Construction)     │
│    对话 Agent 根据上下文生成检索 query │
│    + 自动附加身份过滤条件             │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│ 2. 知识边界过滤 (Knowledge Gate)     │
│    当前模拟的是哪个阶段的角色?        │
│    → 过滤掉角色"不应该知道"的信息     │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│ 3. 检索 + 重排序 (Retrieve & Rank)  │
│    向量检索 → 元数据过滤 → 相关性排序 │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│ 4. 上下文注入 (Context Injection)    │
│    将检索结果作为"角色记忆"注入 prompt │
│    格式: "你记得: {retrieved_content}" │
└─────────────────────────────────────┘
```

**知识边界过滤 (Knowledge Gate) 机制**:

这是运行时 RAG 最关键的设计点。每条索引记录需要额外标注:

```yaml
# 在现有 payload 基础上增加
knowledge_scope:
  known_by: ["辉夜", "八千代", "彩叶"]   # 哪些角色知道这件事
  known_after: "arc_phase_2"              # 从哪个阶段开始知道
  is_secret: false                        # 是否为角色刻意隐瞒的信息
  perspective_bias: "八千代"              # 如果有主观视角偏差, 标注来源角色
```

运行时, 对话 Agent 的当前配置 (模拟哪个角色的哪个阶段) 会自动生成过滤条件:
- 模拟"辉夜" → 过滤掉 `known_by` 不包含"辉夜"的条目
- 模拟"故事前期的彩叶" → 过滤掉 `known_after` 晚于当前阶段的条目
- 模拟"八千代" → 允许访问辉夜的记忆 (因为八千代 = 8000年后的辉夜), 但回忆方式不同

**运行时 vs 构建时的检索差异**:

| 维度 | 构建时 (合成 Agent) | 运行时 (对话 Agent) |
|------|-------------------|-------------------|
| 访问范围 | 全量, 无限制 | 受知识边界过滤 |
| 检索频率 | 密集, 每节多次 | 按需, 仅当对话涉及具体细节时 |
| 延迟要求 | 无严格要求 | < 500ms |
| 结果用途 | 写入 SOUL.md | 注入对话上下文, 以角色口吻转述 |
| 元数据过滤 | 按角色/来源 | 按角色 + 阶段 + 知识边界 |

---

## 3. Phase 3: 对齐层 (Cross-Source Alignment)

### 3.1 对齐策略

```
影片场景列表                   小说场景列表
[fs_001, fs_002, ...]         [ns_001, ns_002, ...]
         │                              │
         └──── 粗筛: Embedding 相似度 ───┘
                        │
                        ▼
              候选对 (相似度 > 阈值)
                        │
                        ▼
              精确判断: LLM Agent
              (基于角色集合/地点/事件三元组)
                        │
                        ▼
              alignment_map.json
```

### 3.2 alignment_map.json 格式

```json
{
  "alignments": [
    {
      "film_scenes": ["fs_012"],
      "novel_scenes": ["ns_008", "ns_009"],
      "confidence": 0.92,
      "notes": "影片合并了小说中两段对话"
    },
    {
      "film_scenes": ["fs_025"],
      "novel_scenes": [],
      "confidence": 1.0,
      "notes": "影片原创场景, 小说中不存在"
    }
  ],
  "unmatched_film": ["fs_078", "fs_079"],
  "unmatched_novel": ["ns_045"]
}
```

### 3.3 处理原则

- 1:1 对齐: 合并两侧的 character_notes, 标注来源差异
- 1:N / N:1: 保留完整映射关系, 合成时需注意信息密度差异
- 未对齐项: 保留为独立条目, **不要丢弃** — 差异本身反映角色在不同媒介中的差异塑造
- 对齐结果需人工抽查 (建议抽查 10-20%)

---

## 4. Phase 4: 合成层 (Synthesis Agent)

### 4.1 Agent 工具集

```python
# 合成 Agent 可调用的工具
tools = [
    "search_memory(query, filters, top_k)",      # 语义搜索索引库
    "get_raw_segment(segment_id)",                # 获取完整原始内容
    "get_all_character_notes(character)",          # 获取角色全部场景笔记
    "get_character_dialogues(character)",          # 获取角色全部原文台词
    "get_alignment_map(character)",                # 跨媒介对应关系
    "read_current_draft()",                        # 读取当前 SOUL.md 草稿
    "write_section(section_name, content)",        # 写入/更新某个章节
    "run_consistency_check()",                     # 检查各章节间一致性
]
```

### 4.2 Agent 工作流

```
1. 初始化: 加载目标角色的全部 character_notes (按时间线排序)
2. 全局扫描: 快速阅读所有笔记, 形成角色整体印象

3. 逐节构建:
   for section in SOUL_MD_TEMPLATE:
       a. 从 character_notes 中检索与该维度最相关的条目
       b. 如需更多细节, 用 search_memory 做定向查询
       c. 如需原始台词/画面, 用 get_raw_segment 回溯
       d. 撰写该章节内容
       e. write_section(section, content)

4. 一致性审查:
   a. run_consistency_check() — 检查各节之间是否矛盾
   b. 特别检查: 负面约束 vs 行为模式 是否一致
   c. 特别检查: 语言风格描述 vs 台词样本 是否匹配
   d. 修正不一致之处

5. 输出最终 SOUL.md
```

### 4.3 模型选择

- 合成 Agent backbone: **Claude Opus** (需要深度推理和长上下文)
- 一致性审查可用同一模型或独立实例 (避免自我确认偏差, 可考虑换模型)

### 4.4 复合身份角色技术适配

在角色级提取 (§1.4) 中, 对复合身份角色需要额外标注:

```yaml
character_scene_note:
  character: "辉夜/八千代"        # 使用统一角色ID
  active_persona: "辉夜"          # 本场景中活跃的身份
  scene_id: "s_042"
  # ... 其余字段不变 ...
  
  # 新增: 跨身份信号
  cross_persona_signal:
    shared_core_evidence: "本场景中体现了跨身份共享内核的什么特征"
    transformation_evidence: "本场景中体现了什么变化/保留信号"
    knowledge_boundary_note: "本场景中角色展示了知道/不知道什么"
```

记忆系统需要支持按 persona 过滤:

```python
# 运行时检索示例: 模拟八千代时
results = memory.search_by_text(
    query="和彩叶第一次在月读直播",
    filters={
        "character": "辉夜/八千代",
        # 八千代可以访问辉夜的记忆 (因为她经历过), 但反过来不行
        "active_persona": {"$in": ["辉夜", "八千代"]},
        "knowledge_accessible_by": "八千代"
    }
)

# 运行时检索示例: 模拟辉夜时 (严格知识隔离)
results = memory.search_by_text(
    query="八千代是谁",
    filters={
        "knowledge_accessible_by": "辉夜"
        # 这会过滤掉所有"八千代=辉夜"的相关信息
    }
)
```

---

## 5. Phase 5: 运行时分发 — MCP Server 架构

> **核心设计决策**: Yorishiro 不自己做对话 agent, 而是产出两个可插拔的制品:  
> 1. **SOUL.md** (markdown 文件) — 宿主应用加载为 system prompt  
> 2. **Yorishiro MCP Server** — 将记忆系统暴露为标准 MCP 工具  
>  
> 任何支持 MCP 的宿主 (Claude Desktop, Cursor, Claude Code, 自建 client) 都能直接使用。

**为什么选 MCP 而不是自建对话 Agent**:

| 维度 | 自建对话 Agent | MCP Server + 宿主 |
|------|--------------|-------------------|
| 模型选择 | 锁定在你选的 LLM | 宿主决定, 用户可选任意模型 |
| UI/UX | 需要自建界面 | 复用成熟产品 (Claude Desktop 等) |
| 维护成本 | 需要维护对话管理、流式输出等 | 只维护工具逻辑, 协议层由 MCP SDK 处理 |
| 可组合性 | 独立系统, 难以与其他工具组合 | 天然与文件系统、浏览器、其他 MCP 服务组合 |
| 分发 | 需要部署完整服务 | `npx yorishiro-mcp` 或 `uvx yorishiro serve` 即可 |

### 5.1 整体运行时架构

```
┌────────────────────────────────────────────────────────────┐
│                MCP 宿主 (如 Claude Desktop)                  │
│                                                            │
│  System Prompt:                                            │
│  ┌────────────────────────────────┐                        │
│  │ SOUL.md (角色人格)             │ ← yorishiro build 产出  │
│  │ + 运行时指令 (persona/阶段)    │                        │
│  └────────────────────────────────┘                        │
│                                                            │
│  用户消息 ──→ LLM ──→ 角色回复                              │
│                │                                           │
│                │ (LLM 自行决定何时调用)                      │
│                ▼                                           │
│  ┌────────────────────────────────────┐                    │
│  │     Yorishiro MCP Server           │                    │
│  │                                    │                    │
│  │  Tools:                            │                    │
│  │   recall_memory(query, persona)    │                    │
│  │   lookup_dialogue(character, kw)   │                    │
│  │   get_world_setting(topic)         │                    │
│  │   list_characters()                │                    │
│  │   get_timeline(character)          │                    │
│  │                                    │                    │
│  │  Prompts:                          │                    │
│  │   character_prompt(name, persona)  │ → 返回完整 system   │
│  │                                    │   prompt 供宿主加载 │
│  │  Resources:                        │                    │
│  │   soul://辉夜/soul.md              │ → 静态 SOUL.md      │
│  │   soul://八千代/soul.md            │                    │
│  └──────────┬─────────────────────────┘                    │
│             │                                              │
└─────────────┼──────────────────────────────────────────────┘
              │
              ▼
┌────────────────────────────────────┐
│         索引层 (Qdrant)             │
│   场景记忆 │ 台词库 │ 世界观 │ 设定 │
│         ↓ 知识边界过滤 ↓            │
│         filtered_results           │
└────────────────────────────────────┘
```

### 5.2 MCP 工具定义

```python
# mcp_server/tools.py

@tool(name="recall_memory")
async def recall_memory(
    query: str,           # 自然语言检索 (如 "和彩叶第一次直播")
    persona: str = None,  # 当前模拟的身份 (用于知识边界过滤)
    arc_phase: str = None # 当前时间线阶段 (进一步过滤)
) -> str:
    """从角色记忆中检索与当前话题相关的场景、事件、细节。
    当对话涉及具体经历、事件细节时使用。
    日常闲聊和性格相关的回应不需要调用此工具。"""

@tool(name="lookup_dialogue")
async def lookup_dialogue(
    character: str,       # 角色名
    keyword: str = None,  # 台词关键词
    scene_id: str = None  # 特定场景
) -> str:
    """查找角色的原文台词样本。
    当需要回忆自己或他人说过的具体话语时使用。"""

@tool(name="get_world_setting")
async def get_world_setting(
    topic: str            # 如 "月读空间规则", "月球社会结构"
) -> str:
    """查询世界观设定细节。
    当对话涉及世界规则、地点、机制等设定信息时使用。"""

@tool(name="get_timeline")
async def get_timeline(
    character: str        # 角色名
) -> str:
    """获取角色的完整时间线和关键事件节点。
    当需要确认事件先后顺序时使用。"""

@tool(name="list_characters")
async def list_characters() -> str:
    """列出当前作品中所有已建档的角色及其关系概要。"""
```

### 5.3 MCP Prompts (预置 System Prompt)

MCP 的 Prompts 功能允许服务端向宿主提供预置的 prompt 模板。Yorishiro 利用这个特性:

```python
# mcp_server/prompts.py

@prompt(name="character_prompt")
async def character_prompt(
    character: str,       # "辉夜" 或 "八千代"
    persona: str = None,  # 复合身份角色的具体 persona
    arc_phase: str = None # 时间线位置
) -> list[PromptMessage]:
    """加载角色扮演的完整 system prompt。
    
    返回: SOUL.md 内容 + 运行时指令 + 工具使用引导。
    宿主应用将此设为 system prompt 即可开始角色对话。"""
    
    soul_md = load_soul_md(character, persona)
    runtime_config = build_runtime_config(persona, arc_phase)
    tool_guide = build_tool_guide()
    
    return [PromptMessage(
        role="user",
        content=f"{soul_md}\n\n{runtime_config}\n\n{tool_guide}"
    )]
```

**用户使用流程** (以 Claude Desktop 为例):

```
1. 配置 claude_desktop_config.json:
   {
     "mcpServers": {
       "yorishiro": {
         "command": "uvx",
         "args": ["yorishiro", "serve", "--project", "./超时空辉夜姬"]
       }
     }
   }

2. 在 Claude Desktop 中:
   - 选择 Yorishiro 提供的 prompt: "character_prompt(八千代)"
   - 开始对话, LLM 自动以八千代的人格回应
   - 当涉及具体细节时, LLM 自动调用 recall_memory 等工具
```

### 5.4 MCP Resources (静态资源暴露)

```python
# mcp_server/server.py

# 将 SOUL.md 和关键配置作为 MCP Resource 暴露
# 宿主可以直接读取这些资源

@resource("soul://{character}/soul.md")
async def get_soul_md(character: str) -> str:
    """角色的 SOUL.md 完整文档"""

@resource("soul://{character}/timeline.json")
async def get_timeline_data(character: str) -> str:
    """角色的时间线结构化数据"""

@resource("soul://meta/characters.json")
async def get_characters_list() -> str:
    """当前项目的所有角色列表及关系概要"""
```

### 5.5 知识边界过滤 (Knowledge Gate) 实现

知识边界过滤在 MCP Server 内部执行, 对宿主透明:

```python
# mcp_server/knowledge_gate.py

class KnowledgeGate:
    def __init__(self, persona: str, arc_phase: str):
        self.persona = persona
        self.arc_phase = arc_phase
    
    def filter(self, results: list[SearchResult]) -> list[SearchResult]:
        """过滤掉当前 persona 在当前阶段不应知道的信息"""
        return [
            r for r in results
            if self.persona in r.metadata.get("known_by", [])
            and self._phase_check(r.metadata.get("known_after"))
        ]
    
    def reframe(self, result: SearchResult) -> str:
        """如果信息的原始视角不是当前 persona, 进行视角转换提示
        例: 辉夜的记忆被八千代回忆时, 添加 "这是你很久以前的记忆" 前缀"""
        if result.metadata.get("active_persona") != self.persona:
            return f"[久远的记忆] {result.content}"
        return result.content
```

### 5.6 RAG 触发策略: 由 LLM 自主决策

在 MCP 架构下, 不再需要单独的 RAG Router。LLM 本身就是路由器——它根据对话上下文自主决定是否调用工具。SOUL.md 中的模拟指令部分提供引导:

```markdown
## 记忆系统使用引导 (写入 SOUL.md 尾部)

你有以下工具可用来回忆细节:
- recall_memory: 回忆具体经历和事件
- lookup_dialogue: 回忆具体说过的话
- get_world_setting: 确认世界规则和设定

使用原则:
- 关于你"是谁"的问题 → 直接回答, 不需要工具
- 关于你"经历过什么"的具体问题 → 用 recall_memory
- 日常闲聊 → 不需要工具, 保持自然
- 不确定的细节 → 宁可用工具确认, 也不要编造
```

### 5.7 多宿主兼容性

| 宿主 | SOUL.md 加载方式 | MCP 连接方式 | 备注 |
|------|-----------------|-------------|------|
| Claude Desktop | Prompt 选择 / 手动粘贴 | claude_desktop_config.json | 最自然的体验 |
| Claude Code | `--system-prompt` flag | `.mcp.json` | 适合开发调试 |
| Cursor | Rules 文件 | MCP 设置 | |
| 自建 Client | API system message | MCP SDK 直连 | 完全控制 |
| OpenAI 兼容 | system message | 需适配层 (MCP → function calling) | 见下 |

**非 MCP 宿主适配 (可选)**:

```python
# 将 MCP 工具转为 OpenAI function calling 格式
# 或直接提供 HTTP API 端点

yorishiro serve --mode http --port 8080  # REST API 模式
yorishiro serve --mode mcp              # MCP stdio 模式 (默认)
yorishiro serve --mode mcp-sse          # MCP SSE 模式 (远程)
```

---

## 6. 编排层 (Orchestration)

### 6.1 推荐目录结构

**轻量方案 (推荐起步)**: Python 脚本 + 手写 state machine

```
yorishiro/
├── extract/
│   ├── film_pipeline.py      # 影片处理
│   ├── novel_pipeline.py     # 小说处理
│   ├── artbook_pipeline.py   # 设定集处理
│   └── character_extractor.py # 角色级提取 (通用)
├── index/
│   ├── indexer.py             # 写入向量数据库
│   └── searcher.py            # 检索接口 (构建时 + MCP 运行时共用)
├── align/
│   └── cross_source_aligner.py
├── synthesize/
│   ├── agent.py               # 合成 Agent 主循环
│   ├── tools.py               # Agent 工具实现
│   └── templates.py           # SOUL.md 模板 (含复合身份版)
├── mcp_server/
│   ├── server.py              # MCP Server 入口 (stdio / SSE)
│   ├── tools.py               # MCP 工具定义 (记忆检索/台词查询/...)
│   ├── knowledge_gate.py      # 知识边界过滤
│   └── prompts.py             # MCP Prompts (预置 system prompt 模板)
├── config.yaml                # 全局配置
└── cli.py                     # yorishiro build / yorishiro serve
```

**进阶方案**: 如果 pipeline 变复杂, 引入 LangGraph 管理 agent 循环中的状态转移和分支逻辑。

### 6.2 不推荐

- 过重的 DAG 框架 (Airflow 等) — 这不是大规模数据工程, 复杂性在 prompt 不在编排
- 过度封装的 agent 框架 (AutoGen, CrewAI 等) — 隐藏了你需要精细控制的细节

---

## 附录 A: 关键 Prompt 模板（骨架）

### A.1 场景级角色提取 Prompt

```
你是一个专业的角色分析师。给定以下场景内容, 请针对角色 [{character_name}] 提取结构化信息。

## 场景内容
{scene_content}

## 提取要求
请以 JSON 格式输出, 包含以下字段:
- dialogue_samples: 该角色在本场景中的原文台词 (保留原文, 不要改写)
- language_traits: 观察到的语言特征
- emotional_state: 当前情绪状态
- inferred_motivation: 推断的行为动机
- internal_conflict: 如有内心冲突
- relationships: [{target, dynamic, shift}]
- actions_taken: 采取的行动
- actions_avoided: 有机会做但选择不做的事 (重要!)
- decision_logic: 决策背后的逻辑
- arc_marker: 是否为角色发展关键节点, 简述

如果该角色未在场景中出现, 只输出: "NOT_PRESENT"
```

### A.2 合成 Agent System Prompt（骨架）

```
你是一个角色心理学专家, 正在为 [{character_name}] 构建 SOUL.md 灵魂文档。

你的目标是生成一份能让 AI Agent 精准模拟该角色人格的参考文档。

你有以下工具可用:
{tool_descriptions}

工作流程:
1. 先用 get_all_character_notes 获取全部场景笔记, 形成整体印象
2. 按 SOUL.md 模板逐节撰写
3. 每写一节前, 先检索相关素材确保有据可依
4. 语言风格部分必须引用原文台词样本
5. 负面约束部分要特别关注 actions_avoided 数据
6. 完成后运行一致性检查

质量标准:
- 每个断言都能追溯到至少一个原始场景
- 语言风格描述必须附带台词样本佐证
- 负面约束必须具体, 不能泛泛而谈
- 角色矛盾面必须被保留, 不要简化为扁平人格
```

### A.3 运行时对话 Agent System Prompt（骨架）

```
{SOUL.md 完整内容 — 或当前 persona 版本}

---

## 运行时配置

当前模拟身份: {persona_name}
时间线位置: {arc_phase}
对话语言: {language}

## 记忆系统使用规则

你有一个记忆系统可以查询。当对话涉及以下情况时, 使用 search_memory 工具:
- 用户问到你经历过的具体事件细节
- 需要回忆具体台词、场景、人物
- 涉及世界观设定的具体规则

不要对每条消息都查询。日常对话、情感表达、性格相关的回应直接基于你的人格回答。

## 知识边界

你知道的: {known_facts_summary}
你不知道的: {unknown_facts_summary}
如果被问到你不知道的事, 以符合你性格的方式表达困惑或好奇, 不要编造。

{if 复合身份角色}
## 身份规则
你当前是{persona_name}。
{cross_persona_rules}
{endif}
```

---

## 附录 B: 辉夜/八千代 复合身份示例（骨架）

> 以《超时空辉夜姬》为例, 展示复合身份 SOUL.md 的关键章节。

### B.1 共享内核示例

```markdown
## 共享内核 — 辉夜与八千代的不变之处

### 根本价值观
- 自由高于一切: 无论是逃离月球的单调工作还是在月读空间创造自由的创作平台
- 连接的渴望: 与他人建立真实的羁绊, 而非隔着屏幕的距离

### 最深层动机
- 与彩叶重逢 / 陪伴彩叶 (这是8000年不变的锚点)

### 核心记忆锚点
- 第一次听到彩叶的歌声
- 被迫返回月球时的无力感
- (八千代独有但源于辉夜时期) 在8000年中反复回忆与彩叶共度的时光
```

### B.2 身份差异对照示例

```markdown
| 维度 | 辉夜 | 八千代 |
|------|------|-------|
| 情绪表达 | 直接爆发, 喜怒形于色 | 内敛温和, 偶尔流露深沉的感伤 |
| 对彩叶 | "最喜欢彩叶了!" (直球) | 默默守护, 不敢贸然相认 |
| 语言风格 | 活泼、口语化、感叹句多 | 优雅、从容, 偶尔有超越年龄的沧桑感 |
| 面对困难 | 冲动行动, 后果再说 | 深思熟虑, 已见过太多兴衰 |
| 孤独处理 | 不理解孤独, 天然向人靠近 | 深谙孤独, 在8000年中与之共处 |
```

### B.3 变化桥梁示例

```markdown
## 变化桥梁: 从辉夜到八千代

### 8000年改变了什么
- 耐心: 从"等不了一秒"到"等了8000年"
- 对时间的感知: 不再认为"现在"理所当然
- 表达方式: 从直接倾泻到克制内敛 (因为她已经知道了离别的痛)
- 身份认知: 学会了隐藏自己, 以AI歌姬的身份存在

### 内在连续性信号 (Agent 可用)
- 当八千代在极度开心时, 会短暂露出辉夜式的孩子气笑容
- 对音乐的本能热爱从未改变 (辉夜的第一次直播 → 八千代的歌姬身份)
- 八千代偶尔会用辉夜时期的口头禅, 然后迅速收敛
```

---

*本文档为活文档，随 pipeline 迭代持续更新。产品需求与路线图见 [requirements.md](./requirements.md)。*
