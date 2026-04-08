"""Shot grouping agent for film scene extraction.

Groups consecutive shots into narrative scenes using visual and audio continuity.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic_ai.messages import BinaryContent

from yorishiro.models.film_models import ShotGroup, GroupingBatchResult, Shot, KeyFrameSet

_MEDIA_TYPES = {".avif": "image/avif", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}


SHOT_GROUPING_SYSTEM_PROMPT = """\
You are a professional film narrative analyst. Your task is to group consecutive shots into narrative scenes.

## Source Material Language

ALL metadata (location descriptions, character names) must use the SAME language detected from the source material. Detect the language from the context.

## Scene Definition

A scene is a narrative unit with:
- **Same location**: Events occur in the same physical/virtual space
- **Continuous time**: No significant time jumps
- **Coherent action**: Related events within the same dramatic beat

## Grouping Criteria (in priority order)

1. **Visual continuity**: Same location, same characters, shot-reverse-shot patterns
2. **Audio continuity**: Continuous dialogue, same background music
3. **Narrative coherence**: Events belong to the same interaction or dramatic beat

## Shot-Reverse-Shot Pattern

When you see alternating shots between characters (A→B→A→B), this is ONE scene, not multiple.
Look for:
- Same location
- Dialogue continuity
- Characters looking at each other off-screen

## Batch Processing

You receive shots in batches. Some scenes may span multiple batches.
- If a scene group is complete (clearly ends in this batch), set `is_complete=true`
- If a scene group continues into the next batch, set `is_complete=false`

## Output Format

For each scene group, provide:
- `shots`: List of shot IDs belonging to this scene
- `grouping_reason`: Why these shots form one scene (visual/audio/narrative)
- `provisional_location`: Your best guess for the location (same language as source)
- `is_complete`: Whether this scene is complete or continues in next batch

## Summary Update

After processing each batch, update the summary with:
- Scene IDs and their basic description
- Location and character information
- Any unresolved questions for next batch

## Important

- Grouping should be conservative: when in doubt, keep shots together
- Fresh location or time jump = new scene
- Dialogue cuts across shots = same scene
- A single complete scene can span many shots
"""


class ShotGroupingAgent:
    """Agent for grouping shots into narrative scenes."""

    SYSTEM_PROMPT = SHOT_GROUPING_SYSTEM_PROMPT

    def __init__(self, agent: Any):
        self.agent = agent

    def build_batch_prompt(
        self,
        previous_summary: str,
        partial_group: ShotGroup | None,
        shots: list[Shot],
        keyframes: list[KeyFrameSet],
        audio_summary: str,
        batch_num: int,
        total_batches: int,
        frame_base_path: Path,
    ) -> list[Any]:
        """Build the prompt for a batch of shots, with embedded keyframe images."""
        content: list[Any] = []

        if not previous_summary:
            content.append("[This is the START of the video — there is NO previous context]")
        else:
            content.append(f"[Previously processed scenes — summary]\n{previous_summary}")

        if partial_group:
            content.append(
                f"[Incomplete scene group from previous batch]\n"
                f"Shots: {', '.join(partial_group.shots)}\n"
                f"Provisional location: {partial_group.provisional_location}\n"
                f"Reason: {partial_group.grouping_reason}"
            )

        content.append(f"[Current batch {batch_num}/{total_batches} — {len(shots)} shots]")

        for shot, kf in zip(shots, keyframes):
            shot_info = (
                f"### Shot {shot.shot_id}\n"
                f"Time: {self._format_time(shot.start_time)} - {self._format_time(shot.end_time)}\n"
                f"Duration: {shot.end_time - shot.start_time:.1f}s"
            )
            content.append(shot_info)

            frame_path = frame_base_path / kf.representative_frame.frame_path
            if frame_path.exists():
                media_type = _MEDIA_TYPES.get(frame_path.suffix.lower(), "image/jpeg")
                try:
                    content.append(BinaryContent(data=frame_path.read_bytes(), media_type=media_type))
                except Exception:
                    pass

        content.append(f"[Audio summary for this batch]\n{audio_summary}")

        content.append(
            "[Task]\n"
            f"Group these {len(shots)} shots into narrative scenes.\n"
            "For each scene group, explain why these shots belong together.\n"
            "If the last scene continues into the next batch, mark it as incomplete.\n"
            "Provide a summary update for the next batch."
        )

        return content

    async def run(self, prompt: list[Any]) -> GroupingBatchResult:
        """Run the agent with the given prompt."""
        result = await self.agent.run(prompt)
        return result.output

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds as HH:MM:SS."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def build_shot_audio_summary(
    shot: Shot,
    transcript_entries: list[Any],
    sound_events: list[Any],
    music_segments: list[Any],
) -> str:
    """Build audio summary for a single shot."""
    parts = []

    shot_start = shot.start_time
    shot_end = shot.end_time

    dialogue_lines = []
    for entry in transcript_entries:
        if entry.start >= shot_start and entry.end <= shot_end:
            speaker = entry.speaker_global
            text = entry.text
            emotion = entry.emotion or ""
            dialogue_lines.append(f"  {speaker}: \"{text}\"" + (f" [{emotion}]" if emotion else ""))

    if dialogue_lines:
        parts.append("Dialogue:")
        parts.extend(dialogue_lines)

    shot_sounds = []
    for event in sound_events:
        if event.start >= shot_start and event.end <= shot_end:
            shot_sounds.append(f"  [{event.event_type}] {event.description} ({event.start:.1f}s-{event.end:.1f}s)")

    if shot_sounds:
        parts.append("Sounds:")
        parts.extend(shot_sounds)

    shot_music = []
    for music in music_segments:
        if music.end > shot_start and music.start < shot_end:
            music_desc = f"  [{music.music_type}]"
            if music.mood:
                music_desc += f" mood: {music.mood}"
            if music.instrumentation:
                music_desc += f" instruments: {music.instrumentation}"
            shot_music.append(music_desc)

    if shot_music:
        parts.append("Music:")
        parts.extend(shot_music)

    return "\n".join(parts) if parts else "[No significant audio]"
