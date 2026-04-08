"""Scene analysis agent for film scene extraction.

Analyzes merged scenes and generates scene content files and metadata.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic_ai.messages import BinaryContent

from yorishiro.models.film_models import (
    FilmSceneContent,
    ShotGroup,
    ShotList,
    Frame,
    Transcript,
    SoundEvent,
    MusicSegment,
)
from yorishiro.audio.speaker_bank import SpeakerBankManager

_MEDIA_TYPES = {".avif": "image/avif", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}


SCENE_ANALYSIS_SYSTEM_PROMPT = """\
You are a professional film analyst. Your task is to analyze film scenes and generate structured scene descriptions.

## Source Material Language

ALL content must be written in the SAME language as the source material. Detect the language from the dialogue and context.

## Your Task

Analyze the provided scene information and produce:

### 1. Scene Content

A complete scene description file with these sections:

#### [视觉描述] (Visual Description)
A comprehensive visual description of the entire scene. Do NOT describe frame-by-frame. Instead, synthesize:
- Setting and atmosphere
- Character positions and movements
- Key visual motifs
- Lighting and color palette
- Important visual details

Write in full sentences, same language as source.

#### [音乐] (Music)
Background music or insert song information:
- Type: BGM or Insert Song
- Mood and emotional quality
- Instrumentation
- If song: lyrics excerpt (first few lines)

#### [对话与声音] (Dialogue and Sounds)
Chronologically ordered events, mixing:
- Dialogue lines with speaker, emotion, and text
- Non-vocal sounds: [非语音: description]
- Ambient sounds: [环境音: description]
- Silence: [长时间沉默]

Timestamp each entry as HH:MM:SS-MM:MM:SS

Format:
`HH:MM:SS Speaker [emotion, volume]: "Dialog text"`

### 2. Character Information

- `characters_present`: Names of characters visually appearing
- `characters_mentioned`: Names mentioned but not visually present
- `speaker_map_update`: Map of SPKR_XXX to character names based on visual confirmation

### 3. Scene Metadata

- `location`: Specific location within the work's world
- `time_of_day`: Morning, afternoon, evening, night, etc.

## Speaker Identification

You will receive a `speaker_map` showing current SPKR_XXX → name mappings.
- If you can visually confirm a speaker is a specific character, update the map
- Be conservative: only confirm when you see the speaker on screen

## Important

- Visual description should be synthesized, not frame-by-frame
- Dialogue timestamps should match the audio data
- All text in the source language
- Be precise about timestamps
"""


class SceneAnalysisAgent:
    """Agent for analyzing merged scenes."""

    SYSTEM_PROMPT = SCENE_ANALYSIS_SYSTEM_PROMPT

    def __init__(self, agent: Any):
        self.agent = agent

    def build_analysis_prompt(
        self,
        scene_group: ShotGroup,
        shot_list: ShotList,
        selected_frames: list[Frame],
        transcript: Transcript,
        sound_events: list[SoundEvent],
        music: list[MusicSegment],
        speaker_bank_manager: SpeakerBankManager,
        project_name: str,
        scene_id: str,
        frame_base_path: Path,
    ) -> list[Any]:
        """Build the prompt for scene analysis, with embedded keyframe images."""
        content: list[Any] = []

        start_shot = shot_list.get_shot(scene_group.shots[0])
        end_shot = shot_list.get_shot(scene_group.shots[-1])

        scene_start = start_shot.start_time if start_shot else 0.0
        scene_end = end_shot.end_time if end_shot else 0.0

        header = "\n".join([
            f"# Scene Analysis: {scene_id}",
            f"Project: {project_name}",
            f"Time range: {self._format_time(scene_start)} - {self._format_time(scene_end)}",
            f"Shots: {', '.join(scene_group.shots)}",
            f"Provisional location: {scene_group.provisional_location}",
            "",
            "## Key Frames",
        ])
        content.append(header)

        for frame in selected_frames:
            frame_path = frame_base_path / frame.frame_path
            if frame_path.exists():
                media_type = _MEDIA_TYPES.get(frame_path.suffix.lower(), "image/jpeg")
                content.append(f"Frame at {self._format_time(frame.timestamp)}:")
                try:
                    content.append(BinaryContent(data=frame_path.read_bytes(), media_type=media_type))
                except Exception:
                    pass

        transcript_lines = ["\n## Transcript\n"]
        for entry in transcript.entries:
            if entry.end > scene_start and entry.start < scene_end:
                speaker_name = speaker_bank_manager.get_speaker_name(entry.speaker_global) or entry.speaker_global
                emotion_str = f" [{entry.emotion}]" if entry.emotion else ""
                volume_str = f", {entry.volume}" if entry.volume else ""
                transcript_lines.append(
                    f"{self._format_time(entry.start)} {speaker_name}{emotion_str}{volume_str}: \"{entry.text}\""
                )

        sound_lines = ["\n## Sound Events\n"]
        for event in sound_events:
            if event.end > scene_start and event.start < scene_end:
                sound_lines.append(
                    f"{self._format_time(event.start)}-{self._format_time(event.end)}: [{event.event_type}] {event.description}"
                )

        music_lines = ["\n## Music\n"]
        for seg in music:
            if seg.end > scene_start and seg.start < scene_end:
                music_desc = f"{self._format_time(seg.start)}-{self._format_time(seg.end)}: {seg.music_type}"
                if seg.mood:
                    music_desc += f", {seg.mood}"
                if seg.instrumentation:
                    music_desc += f", instruments: {seg.instrumentation}"
                music_lines.append(music_desc)
                if seg.has_lyrics and seg.lyrics_excerpt:
                    music_lines.append(f"  Lyrics: {seg.lyrics_excerpt}")

        speaker_lines = ["\n## Current Speaker Map\n"]
        for spk_info in speaker_bank_manager.speaker_bank.speakers:
            status = "confirmed" if spk_info.confidence == "confirmed" else "unconfirmed"
            name = speaker_bank_manager.speaker_bank.speaker_map.get(spk_info.speaker_id, "?")
            speaker_lines.append(f"  {spk_info.speaker_id} → {name} [{status}]")

        task_text = "\n".join([
            "\n## Task\n",
            "Analyze this scene and provide:",
            "1. Visual description (synthesized, not frame-by-frame)",
            "2. Music description",
            "3. Chronological dialogue and sounds",
            "4. Character list (present and mentioned)",
            "5. Confirmed speaker identities (only if visually confirmed)",
            "6. Location and time of day",
            "\nAll text content must be in the source language (same as dialogue).",
        ])

        content.append("\n".join(transcript_lines + sound_lines + music_lines + speaker_lines))
        content.append(task_text)

        return content

    async def run(self, prompt: list[Any]) -> FilmSceneContent:
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

    def format_scene_file(
        self,
        scene_id: str,
        source_file: str,
        scene_start: float,
        scene_end: float,
        content: FilmSceneContent,
    ) -> str:
        """Format the scene content as a text file."""
        lines = [
            f"# scene_{scene_id}.txt",
            f"# Source: {source_file}",
            f"# Scene ID: {scene_id}",
            f"# Timestamp: {self._format_time(scene_start)} - {self._format_time(scene_end)}",
            f"# Location: {content.location}",
            f"# Characters: {', '.join(content.characters_present)}",
            "# ---",
            "",
            "[视觉描述]",
            content.visual_description,
            "",
            "[音乐]",
            content.music,
            "",
            "[对话与声音]",
            content.dialogue_and_sounds,
        ]
        return "\n".join(lines)
