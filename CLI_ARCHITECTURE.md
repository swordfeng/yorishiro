# Yorishiro CLI 工具架构

> 每个步骤都是独立的命令行工具，通过 YAML Frontmatter 传递数据。

## 设计理念

1. **独立运行**：每个工具都可以单独执行，不依赖全局环境
2. **函数可复用**：CLI 只是函数的 wrapper，核心函数可以被其他代码调用
3. **标准格式**：使用 YAML Frontmatter（`---\nmetadata\n---\ncontent`）传递数据
4. **配置来源**：最终从 `material.yaml` 读取，但单个脚本可以独立运行

## 工具链

```
extract_chapters.py      # EPUB → chXXX.txt (YAML frontmatter)
segment_scenes.py        # chXXX.txt → scene_XXX.txt (YAML frontmatter)  
extract_character.py     # scene_XXX.txt → character_notes.yaml
```

## YAML Frontmatter 格式

标准格式（被 Jekyll、Hugo 等广泛使用）：

```yaml
---
index: 0
title: "第一章 降临"
length: 15234
source_file: "CPK.epub"
---
这是正文内容...
可以有多行...
```

**解析方式**：
```python
import yaml

def parse_frontmatter(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 分割 frontmatter 和正文
    if content.startswith('---'):
        parts = content.split('---', 2)
        if len(parts) >= 3:
            metadata = yaml.safe_load(parts[1])
            body = parts[2].strip()
            return metadata, body
    
    return {}, content
```

## 使用示例

### 独立运行

```bash
# 提取章节
uv run python3 extract_chapters.py \
    --input material/raw/CPK.epub \
    --output material/processed/novel/CPK/chapters

# 切分场景
uv run python3 segment_scenes.py \
    --input material/processed/novel/CPK/chapters/ch000.txt \
    --output material/processed/novel/CPK/scenes/ch000/

# 提取角色
uv run python3 extract_character.py \
    --input material/processed/novel/CPK/scenes/ \
    --character "彩葉" \
    --output material/processed/novel/CPK/characters/彩葉.yaml
```

### 作为函数调用

```python
from extract_chapters import extract_chapters

# 直接调用函数
saved_files = extract_chapters(
    input_path=Path("book.epub"),
    output_dir=Path("chapters/")
)
```

## 从 material.yaml 驱动

最终的 Pipeline 会从 `material.yaml` 读取配置，调用这些工具：

```yaml
# material.yaml
sources:
  - type: "epub"
    path: "material/raw/CPK.epub"
    
pipeline:
  steps:
    - name: "extract_chapters"
      input: "{source.path}"
      output: "material/processed/novel/CPK/chapters"
    
    - name: "segment_scenes"
      input: "material/processed/novel/CPK/chapters"
      output: "material/processed/novel/CPK/scenes"
      config:
        target_tokens_per_scene: 3000
```

Pipeline runner：
```python
# pipeline_runner.py
import yaml
import subprocess

material = yaml.safe_load(open("material.yaml"))

for step in material["pipeline"]["steps"]:
    cmd = [
        "uv", "run", "python3", f"{step['name']}.py",
        "--input", step["input"],
        "--output", step["output"]
    ]
    subprocess.run(cmd, check=True)
```

## 优点

1. **可调试**：每一步都可以单独运行和检查
2. **可复用**：核心函数可以被其他项目使用
3. **透明**：中间产物都是文本文件，易于查看
4. **灵活**：可以跳过某些步骤或重新运行特定步骤

## 下一步

1. 创建 `segment_scenes.py`
2. 创建 `extract_character.py`
3. 创建 `pipeline_runner.py` 从 material.yaml 驱动
