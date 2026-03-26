# Scene Segmentation Learnings | 场景分段经验总结

> Documenting issues and lessons learned from Phase 0 scene segmentation.

---

## Issues Identified | 发现的问题

### 1. Language Inconsistency | 语言不统一

**Problem**: Manifests use mixed languages (Japanese, English, Chinese).

**Examples from ch004**:
```json
// BAD - mixed language
{"location": "路地（street）", "time": "夜（night）"}

// GOOD - source language only (ch003)
{"location": "ゲーム（KASSEN）内・音声チャット", "time": "放課後〜夜"}
```

**Rule**: Always use the source material's language for metadata.

### 2. Over-Segmentation | 过度分割

**Problem**: ch004 has 32 scenes - far too many. Many consecutive scenes have identical location/time.

**Examples**:
- scenes 0-1: both `路地（street）` / `夜（night）` - should be merged
- scenes 4-5: both `アパート（apartment）` / `夜（night）` - should be merged
- scenes 24-25: both `ツクヨミ・ライブ会場` / `夜（night）` - should be merged

**Better segmentation (ch003)**: 9 scenes for a full chapter with multiple locations and time jumps.

**Rule**: Only split on actual boundaries (location change, time jump, POV change, narrative marker). If consecutive scenes have same location/time, they should be one scene.

### 3. Inconsistent Subagent Methods | Subagent 方法不一致

**Problem**: Each subagent interpreted the task differently.

**Root cause**: 
- Instructions were not specific enough
- Subagents had freedom to choose their approach
- No standardized output format enforcement

**Evidence**:
- ch003: Clean 9-scene segmentation with consistent Japanese
- ch004: Chaotic 32-scene over-segmentation with mixed language
- ch005-ch011: Various quality levels

### 4. Shell Tool Usage | 工具使用不当

**Problem**: Some subagents tried to use shell commands (grep, wc, etc.) to process content, which didn't help and added complexity.

**Lesson**: Subagents should read files directly, not try to parse with shell tools.

### 5. No Quality Check | 缺乏质量检查

**Problem**: No final review step after subagent completion.

**Missing process**:
- Main agent should verify output quality before declaring done
- Check for language consistency
- Check for reasonable scene count
- Check for complete content

---

## ch003 vs ch004 Comparison | ch003 与 ch004 对比

| Aspect | ch003 (Good) | ch004 (Bad) |
|--------|-------------|-------------|
| Scene count | 9 | 32 |
| Language | 全日语 | 日英混合 |
| Location consistency | Varied properly | 重复 location |
| Segmentation logic | Clear boundaries | Over-segmented |

---

## Recommendations for Future Subagent Tasks | 未来 Subagent 任务建议

### 1. Strict Language Rule | 严格语言规则

```
System language = source material language
Manifest language = source material language
```

### 2. Scene Count Guidelines | 场景数量指导

- Short chapter (<5000 chars): 2-4 scenes
- Medium chapter (5000-15000 chars): 4-8 scenes
- Long chapter (>15000 chars): 6-12 scenes
- If finding >15 scenes, likely over-segmentation

### 3. Merge Rules | 合并规则

Only split if ALL of these are true:
- Location changed
- OR Time jumped significantly (hours/days)
- OR POV changed
- OR Narrative marker present (※, ──, etc.)

If none of the above, scenes should be merged.

### 4. Quality Checklist | 质量检查清单

After completing segmentation, verify:
- [ ] All metadata in source language
- [ ] Scene count reasonable (per guidelines above)
- [ ] Each scene has unique boundary justification
- [ ] Content preserved without modification
- [ ] Manifest JSON valid

### 5. Subagent Prompt Template | Subagent 提示词模板

```
## Task: Segment Chapter X into scenes

## Source Material Language: [指定语言]
## Output Language: [必须与源语言一致]

## Hard Rules:
1. Output manifest MUST use [指定语言] for all metadata
2. Scene count target: [N] scenes maximum
3. ONLY split on: location change, time jump, POV change, narrative marker
4. If consecutive scenes have same location/time, MERGE them

## Quality Check (MUST do before finishing):
- [ ] Language consistency verified
- [ ] Scene count within guideline
- [ ] All scene files created and verified
```

---

## Files Affected | 受影响的文件

### Need Re-do | 需要重做
- `ch004/` - 32 scenes, mixed language, over-segmented

### Quality OK | 质量尚可
- `ch003/` - 9 scenes, good quality
- `ch005/` - 9 scenes
- `ch006/` - 8 scenes
- `ch007/` - 8 scenes
- `ch008/` - 4 scenes
- `ch009/` - 18 scenes (may be over)
- `ch010/` - 4 scenes
- `ch011/` - 5 scenes

### Skip | 跳过
- `ch000`, `ch001`, `ch002` - non-narrative
- `ch012`-`ch015` - metadata only

---

## Next Steps | 下一步

1. **Re-do ch004** with stricter guidelines
2. **Review ch009** - 18 scenes may be over-segmented
3. **Implement quality check step** in main agent workflow
4. **Update scene_segmentation.py** with better prompts
