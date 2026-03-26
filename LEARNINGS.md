# Learnings | 经验总结

> Documenting issues and lessons learned from Phase 0.

---

## Scene Segmentation Issues | 场景分段问题

### 1. Language Inconsistency | 语言不统一

**Problem**: Manifests use mixed languages (Japanese, English, Chinese).

**Examples**:
```json
// BAD - mixed language
{"location": "路地（street）", "time": "夜（night）"}

// GOOD - source language only
{"location": "ゲーム（KASSEN）内・音声チャット", "time": "放課後〜夜"}
```

**Rule**: Always use the source material's language for all metadata.

### 2. Incorrect Segmentation | 不正确分割

**Problem**: ch004 had scenes with identical consecutive locations/times that should have been merged, and scenes split without proper justification.

**Rule**: Split ONLY on actual boundaries (location change, time jump, POV change, narrative marker). Merge if consecutive scenes share same location/time/POV. The number of scenes is irrelevant - correctness matters, not count.

---

## Prompt Design | Prompt 设计

### 4. Self-Contained Prompts | Prompt 自包含

**Problem**: Prompt referenced ch003 as example, which is not in context.

**Rule**: Prompt should be self-contained. Do not reference external files or examples that subagent cannot access.

### 5. No Original Text in Prompt | 不在 Prompt 中嵌入原文

**Problem**: Embedding chapter content in prompt causes token waste and truncation.

**Rule**: Point to source file instead. Let subagent read the file directly.

### 6. Large File Handling | 大文件处理

**Rule**: If file is too large, read in parts. Track position to ensure no omission or duplication.

### 7. Character Naming in Scene Segmentation | 场景分段中的角色命名

**Problem**: Same character called by different names in different scenes (e.g., "赤ちゃん" → "少女" for Kaguya).

**Solution**: Scene segmentation manifest uses names AS THEY APPEAR in the text. Do NOT normalize names here.

**Rationale**: This is a character extraction concern, not scene segmentation. The alias mapping (identifying "赤ちゃん" and "少女" as the same entity) should be handled in the character extraction phase.

**Rule**: In scene segmentation, `characters` field lists names as they appear in that specific scene. Normalization happens later.

---

## Workflow | 工作流

### 7. Simplified Workflow | 简化工作流

**Old (unnecessary)**:
```
epub → generate prompt files → copy to LLM → subagent execute
```

**New (correct)**:
```
1. Execute python snippet to get prompt
2. Give prompt directly to subagent
3. Subagent reads source file, executes task
```

**Never create workaround scripts for errors** - fix the root cause.

### 8. How to Get Prompt for Subagent | 如何获取 Prompt

```bash
uv run python -c "from extract.scene_segmentation import SYSTEM_PROMPT, build_scene_segmentation_prompt; print(build_scene_segmentation_prompt(4))"
```

Then give the output directly to subagent.

### 9. Code Organization | 代码组织

**Keep minimal scripts**:
- `epub_pipeline.py` - core, epub parsing
- `scene_segmentation.py` - prompt template
- `character_extractor.py` - prompt template

**Delete intermediate/one-time scripts** - they add complexity without value.

---

## Quality Checklist | 质量检查清单

Before finishing any subagent task, verify:
- [ ] All metadata uses source material language
- [ ] Each split has clear boundary justification
- [ ] No text omission or duplication
- [ ] Manifest JSON is valid

---

## Files Status | 文件状态

### Need Re-do | 需要重做
- `ch004/` - incorrect segmentation, needs redo

### Quality OK | 质量尚可
- `ch003/` - good
- `ch005/` - needs review
- `ch006/` - needs review
- `ch007/` - needs review
- `ch008/` - needs review
- `ch009/` - needs review (18 scenes)
- `ch010/` - needs review
- `ch011/` - needs review

### Skip | 跳过
- `ch000`, `ch001`, `ch002` - non-narrative
- `ch012`-`ch015` - metadata only

---

## Next Steps | 下一步

1. Re-do ch004 with improved prompts
2. Review remaining chapters for quality
3. Proceed to character extraction when scene segmentation is verified
