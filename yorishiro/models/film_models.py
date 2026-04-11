"""Pydantic models for film scene extraction pipeline.

Data structures for:
- Layer 0A: Video processing (shots, frames)
- Layer 0B: Audio analysis (transcript, speakers, sound events, music)
- Step 1: Shot grouping (grouping result)
- Step 2: Scene analysis (scene content and metadata)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Layer 0A: Video processing
# ---------------------------------------------------------------------------


class Shot(BaseModel):
    shot_id: str = Field(description="Shot identifier, e.g. 'sh001'")
    start_time: float = Field(description="Start time in seconds")
    end_time: float = Field(description="End time in seconds")
    start_frame: int = Field(description="Start frame number")
    end_frame: int = Field(description="End frame number")


class ShotList(BaseModel):
    video_file: str = Field(description="Source video filename")
    video_hash: str = Field(description="Hash based on mtime + filesize")
    fps: float = Field(description="Frames per second")
    duration: float = Field(description="Total duration in seconds")
    shots: list[Shot] = Field(default_factory=list)

    def get_shot(self, shot_id: str) -> Shot | None:
        for shot in self.shots:
            if shot.shot_id == shot_id:
                return shot
        return None


class Frame(BaseModel):
    frame_path: str = Field(description="Relative path to frame image file")
    timestamp: float = Field(description="Timestamp in seconds")
    frame_number: int = Field(description="Frame number in video")


class KeyFrameSet(BaseModel):
    shot_id: str = Field(description="Shot this belongs to")
    representative_frame: Frame = Field(
        description="Single frame for shot grouping agent (Step 1)"
    )
    extracted_frames: list[Frame] = Field(
        default_factory=list,
        description="All frames extracted from this shot (for scene-level selection)",
    )


# ---------------------------------------------------------------------------
# Layer 0B: Audio analysis
# ---------------------------------------------------------------------------


class SpeakerSegment(BaseModel):
    speaker_id: str = Field(
        description="Local speaker ID from diarization (e.g., 'SP_A')"
    )
    start: float = Field(description="Start time in seconds")
    end: float = Field(description="End time in seconds")


class STTEntry(BaseModel):
    start: float = Field(description="Start time in seconds")
    end: float = Field(description="End time in seconds")
    text: str = Field(description="Transcribed text")
    confidence: float = Field(description="Transcription confidence")


class STTTranscript(BaseModel):
    language: str = Field(description="Detected or configured language")
    entries: list[STTEntry] = Field(default_factory=list)


class WindowVote(BaseModel):
    speaker_id: str = Field(description="Cluster speaker ID for this window")
    similarity: float = Field(
        description="Cosine similarity of window embedding to cluster centroid"
    )
    start: float = Field(description="Window start time in seconds")
    end: float = Field(description="Window end time in seconds")


class SpeakerAttributionEntry(BaseModel):
    entry_id: str = Field(description="Stable STT entry identifier")
    start: float = Field(description="Entry start time in seconds")
    end: float = Field(description="Entry end time in seconds")
    speaker_id: str = Field(description="Attributed global speaker ID or UNKNOWN")
    text: str = Field(default="", description="STT transcribed text for this entry")
    similarity: float | None = Field(
        default=None, description="Best speaker similarity score if available"
    )
    embedding_present: bool = Field(
        description="Whether an embedding was available for attribution"
    )
    enrolled: bool = Field(
        description="Whether this entry was used to update the speaker bank"
    )
    window_votes: list[WindowVote] = Field(
        default_factory=list,
        description="Per-window cluster assignments and similarities",
    )


class SpeakerAttribution(BaseModel):
    entries: list[SpeakerAttributionEntry] = Field(default_factory=list)


class TranscriptEntry(BaseModel):
    speaker_global: str = Field(description="Global speaker ID (e.g., 'SPKR_001')")
    start: float = Field(description="Start time in seconds")
    end: float = Field(description="End time in seconds")
    text: str = Field(description="Transcribed text")
    confidence: float = Field(description="Transcription confidence")
    emotion: str | None = Field(default=None, description="Detected emotion")
    pitch_trend: str | None = Field(default=None, description="Pitch trend description")
    speech_rate: str | None = Field(default=None, description="Speech rate description")
    volume: str | None = Field(default=None, description="Volume level description")


class Transcript(BaseModel):
    language: str = Field(description="Detected or configured language")
    entries: list[TranscriptEntry] = Field(default_factory=list)


class SpeakerInfo(BaseModel):
    speaker_id: str = Field(description="Global speaker ID (e.g., 'SPKR_001')")
    first_seen_time: float = Field(description="First occurrence timestamp")
    first_seen_scene: str | None = Field(
        default=None, description="First scene ID where this speaker appears"
    )
    provisional_name: str | None = Field(
        default=None, description="Provisional name from VLM analysis"
    )
    confidence: str = Field(
        default="unconfirmed", description="'confirmed' or 'unconfirmed'"
    )


class SpeakerClusterQuality(BaseModel):
    speaker_id: str = Field(description="Global speaker ID")
    num_utterances: int = Field(
        description="Number of utterances assigned to this speaker"
    )
    num_windows: int = Field(
        description="Number of post-filter windows in this cluster"
    )
    num_enrolled: int = Field(
        description="Number of utterances with ≥2 window votes (enrolled)"
    )
    mean_similarity: float | None = Field(
        default=None,
        description="Mean cosine similarity to centroid (enrolled utterances only)",
    )
    median_similarity: float | None = Field(
        default=None,
        description="Median cosine similarity to centroid (enrolled utterances only)",
    )
    p25_similarity: float | None = Field(
        default=None, description="25th percentile similarity to centroid (enrolled)"
    )
    p75_similarity: float | None = Field(
        default=None, description="75th percentile similarity to centroid (enrolled)"
    )
    min_similarity: float | None = Field(
        default=None, description="Min similarity to centroid (enrolled)"
    )
    max_similarity: float | None = Field(
        default=None, description="Max similarity to centroid (all utterances)"
    )
    intra_cluster_similarity: float | None = Field(
        default=None,
        description="Mean pairwise cosine similarity among windows within this cluster (post-filter)",
    )
    nearest_speaker: str | None = Field(
        default=None, description="Speaker ID of the most similar other centroid"
    )
    nearest_speaker_similarity: float | None = Field(
        default=None, description="Cosine similarity to nearest other centroid"
    )


class SpeakerBank(BaseModel):
    speakers: list[SpeakerInfo] = Field(default_factory=list)
    cluster_quality: list[SpeakerClusterQuality] = Field(default_factory=list)
    speaker_map: dict[str, str] = Field(
        default_factory=dict,
        description="Maps global speaker ID to character name (when confirmed)",
    )

    def get_speaker(self, speaker_id: str) -> SpeakerInfo | None:
        for spk in self.speakers:
            if spk.speaker_id == speaker_id:
                return spk
        return None

    def add_speaker(self, speaker_id: str, first_seen_time: float) -> SpeakerInfo:
        info = SpeakerInfo(
            speaker_id=speaker_id,
            first_seen_time=first_seen_time,
        )
        self.speakers.append(info)
        return info

    def confirm_speaker(
        self, speaker_id: str, character_name: str, scene_id: str | None = None
    ) -> None:
        for spk in self.speakers:
            if spk.speaker_id == speaker_id:
                spk.provisional_name = character_name
                spk.confidence = "confirmed"
                spk.first_seen_scene = scene_id
                break
        self.speaker_map[speaker_id] = character_name


class SoundEvent(BaseModel):
    start: float = Field(description="Start time in seconds")
    end: float = Field(description="End time in seconds")
    event_type: str = Field(
        description="Event type: 'non_speech_vocal', 'ambient', 'sfx', 'silence'"
    )
    description: str = Field(description="Natural language description of the sound")


class MusicSegment(BaseModel):
    start: float = Field(description="Start time in seconds")
    end: float = Field(description="End time in seconds")
    music_type: Literal["bgm", "insert_song"] = Field(description="Music type")
    has_lyrics: bool = Field(default=False, description="Whether it has lyrics")
    lyrics_excerpt: str | None = Field(
        default=None, description="Lyrics excerpt if available"
    )
    mood: str | None = Field(default=None, description="Mood description")
    instrumentation: str | None = Field(default=None, description="Main instruments")
    bpm: int | None = Field(default=None, description="Beats per minute")
    valence: str | None = Field(
        default=None, description="Valence: 'low', 'medium', 'high'"
    )
    arousal: str | None = Field(
        default=None, description="Arousal: 'low', 'medium', 'high'"
    )


# ---------------------------------------------------------------------------
# Step 1: Shot grouping
# ---------------------------------------------------------------------------


class ShotGroup(BaseModel):
    shots: list[str] = Field(
        description="Shot IDs belonging to this scene, e.g., ['sh011', 'sh012', 'sh013']"
    )
    grouping_reason: str = Field(
        description="Reason for grouping: visual/audio/narrative logic"
    )
    provisional_location: str = Field(description="Preliminary location guess")
    is_complete: bool = Field(
        default=True,
        description="True if complete scene, False if truncated at batch end",
    )


class GroupingBatchResult(BaseModel):
    new_scene_groups: list[ShotGroup] = Field(
        description="Scene groups identified in this batch"
    )
    has_more: bool = Field(description="Whether more shots remain after this batch")
    next_shot_index: int = Field(description="Index of next shot to process")
    summary_update: str = Field(
        description="Summary of processed content, for next batch context"
    )


# ---------------------------------------------------------------------------
# Step 2: Scene analysis
# ---------------------------------------------------------------------------


class FilmSceneContent(BaseModel):
    visual_description: str = Field(
        description="Comprehensive visual description of the scene"
    )
    music: str = Field(description="Music description (BGM or insert song)")
    dialogue_and_sounds: str = Field(
        description="Dialogue and sound events in timestamp order"
    )
    characters_present: list[str] = Field(
        description="Character names seen in this scene"
    )
    characters_mentioned: list[str] = Field(
        default_factory=list, description="Characters mentioned but not seen"
    )
    location: str = Field(description="Scene location")
    time_of_day: str = Field(description="Time of day in scene")
    speaker_map_update: dict[str, str] = Field(
        default_factory=dict,
        description="Speaker ID → character name mappings confirmed in this scene",
    )


class FilmSceneMetadata(BaseModel):
    scene_id: str = Field(description="Scene ID, e.g., 'fs001'")
    source: dict[str, Any] = Field(description="Source info: type, file, shots, times")
    content_file: str = Field(description="Path to scene content file")
    token_estimate: int = Field(description="Estimated token count")
    location: str = Field(description="Scene location")
    time_of_day: str = Field(description="Time of day")
    characters: dict[str, list[str]] = Field(
        description="{'present': [...], 'mentioned': [...]}"
    )
    audio: dict[str, Any] = Field(
        description="Audio summary: has_music, music_type, notable_sounds"
    )
    speaker_map_snapshot: dict[str, str] = Field(
        description="Speaker ID → name mapping at this point in the video"
    )


class FilmSceneIndex(BaseModel):
    metadata: dict[str, Any] = Field(
        description="Version, generated_at, project, source info"
    )
    scenes: list[FilmSceneMetadata] = Field(default_factory=list)

    def save(self, path: Path) -> None:
        import json

        path.write_text(
            json.dumps(self.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> FilmSceneIndex:
        import json

        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(**data)
