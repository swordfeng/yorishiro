"""Music analysis: BGM and insert song detection.

Uses Demucs for music/vocals separation, Essentia for music features.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import librosa
import numpy as np
import soundfile as sf
import torch
from dataclasses import dataclass

from yorishiro.models.film_models import MusicSegment
from yorishiro.utils import get_device


@dataclass
class MusicAnalyzerConfig:
    separation_backend: str = "demucs"
    separation_model: str = "htdemucs"
    analysis_backend: str = "essentia"
    detect_lyrics: bool = True


class MusicAnalyzer:
    """Analyzes music segments in audio."""

    def __init__(self, config: MusicAnalyzerConfig | None = None):
        self.config = config or MusicAnalyzerConfig()
        self._demucs_model = None
        self._essentia = None

    def release_models(self) -> None:
        from yorishiro.audio._speech_support import clear_torch_cache

        self._demucs_model = None
        self._essentia = None
        clear_torch_cache()

    def analyze(
        self,
        video_path: Path,
        audio_path: Path,
        output_dir: Path,
        force: bool = False,
        nonvoice_path: Path | None = None,
    ) -> list[MusicSegment]:
        """Analyze music in video.

        If nonvoice_path is provided (pre-separated non-voice stem), internal
        Demucs separation is skipped and that file is used directly.
        Returns list of MusicSegment.
        Caches to output_dir / music_analysis.json.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        cache_file = output_dir / "music_analysis.json"

        if not force and cache_file.exists():
            try:
                cached = json.loads(cache_file.read_text(encoding="utf-8"))
                return [MusicSegment(**m) for m in cached.get("segments", [])]
            except Exception:
                pass

        if nonvoice_path is not None:
            print(
                f"  [MusicAnalyzer] Using pre-separated non-voice stem: {nonvoice_path.name}"
            )
            music_path, vocals_path = nonvoice_path, None
        else:
            print(f"  [MusicAnalyzer] Separating music from {video_path.name} ...")
            music_path, vocals_path = self._separate_music(audio_path, output_dir)

        print("  [MusicAnalyzer] Detecting music segments ...")
        segments = self._detect_music_segments(music_path, audio_path, vocals_path)

        print("  [MusicAnalyzer] Analyzing music features ...")
        segments = self._analyze_features(music_path, segments)

        result = {"segments": [s.model_dump() for s in segments]}
        cache_file.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        print(f"  [MusicAnalyzer] Found {len(segments)} music segments")
        return segments

    def _separate_music(
        self, audio_path: Path, output_dir: Path
    ) -> tuple[Path, Path | None]:
        """Separate music track using Demucs Python API.

        Returns (music_no_vocals_path, vocals_path). vocals_path is None if separation failed.
        """
        music_path = output_dir / "music_separated.wav"
        vocals_path = output_dir / "vocals_separated.wav"

        if music_path.exists() and vocals_path.exists():
            return music_path, vocals_path

        try:
            from demucs import pretrained
            from demucs.apply import apply_model
            from demucs.audio import AudioFile

            device = get_device()
            model = pretrained.get_model(self.config.separation_model)
            model.to(device)
            model.eval()

            audio_file = AudioFile(audio_path)
            audio = cast(Any, audio_file.read)(streams=0, samplerate=44100, channels=2)

            with torch.no_grad():
                sources = apply_model(model, audio[None].to(device), device=device)
                sources = sources.cpu()

            vocals_idx = model.sources.index("vocals")
            no_vocals = sources[0].sum(0) - sources[0, vocals_idx]
            vocals = sources[0, vocals_idx]

            sf.write(str(music_path), no_vocals.numpy().T, 44100)
            sf.write(str(vocals_path), vocals.numpy().T, 44100)

            return music_path, vocals_path

        except ImportError:
            print("    [MusicAnalyzer] Demucs not installed, using original audio")
            return audio_path, None
        except Exception as e:
            print(f"    [MusicAnalyzer] Demucs error: {e}, using original audio")
            return audio_path, None

    def _detect_music_segments(
        self, music_path: Path, original_path: Path, vocals_path: Path | None = None
    ) -> list[MusicSegment]:
        """Detect segments where music is present, and classify bgm vs insert_song."""
        try:
            y, sr = librosa.load(str(music_path), sr=None)
            duration = len(y) / sr

            hop_length = 512
            frame_length = 2048

            rms = librosa.feature.rms(
                y=y, frame_length=frame_length, hop_length=hop_length
            )[0]

            threshold = np.mean(rms) * 1.5

            is_music = rms > threshold

            segments = []
            in_segment = False
            segment_start = 0.0

            for i, music in enumerate(is_music):
                time = i * hop_length / sr

                if music and not in_segment:
                    in_segment = True
                    segment_start = time
                elif not music and in_segment:
                    segment_duration = time - segment_start
                    if segment_duration > 3.0:
                        segments.append(
                            MusicSegment(
                                start=segment_start,
                                end=time,
                                music_type="bgm",
                                has_lyrics=False,
                            )
                        )
                    in_segment = False

            if in_segment:
                segment_duration = duration - segment_start
                if segment_duration > 3.0:
                    segments.append(
                        MusicSegment(
                            start=segment_start,
                            end=duration,
                            music_type="bgm",
                            has_lyrics=False,
                        )
                    )

            # Classify insert_song vs bgm using vocal energy ratio
            if vocals_path and vocals_path.exists() and segments:
                try:
                    y_vocals, sr_vocals = librosa.load(str(vocals_path), sr=None)
                    sr_vocals = int(sr_vocals)
                    for seg in segments:
                        v_start = int(seg.start * sr_vocals)
                        v_end = int(seg.end * sr_vocals)
                        m_start = int(seg.start * int(sr))
                        m_end = int(seg.end * int(sr))
                        vocal_chunk = y_vocals[v_start:v_end]
                        music_chunk = y[m_start:m_end]
                        if len(vocal_chunk) > 0 and len(music_chunk) > 0:
                            vocal_rms = float(np.sqrt(np.mean(vocal_chunk**2)))
                            music_rms = float(np.sqrt(np.mean(music_chunk**2)))
                            if music_rms > 0 and vocal_rms / music_rms > 0.3:
                                seg.music_type = "insert_song"
                                seg.has_lyrics = True
                except Exception as e:
                    print(f"    [MusicAnalyzer] Vocal classification error: {e}")

            return segments

        except Exception as e:
            print(f"    [MusicAnalyzer] Error detecting segments: {e}")
            return []

    def _analyze_features(
        self, music_path: Path, segments: list[MusicSegment]
    ) -> list[MusicSegment]:
        """Analyze music features for each segment using Essentia."""
        try:
            if not music_path.exists():
                return segments

            y, sr = librosa.load(str(music_path), sr=None)
            sr_int = int(sr)

            for segment in segments:
                start_sample = int(segment.start * sr_int)
                end_sample = int(segment.end * sr_int)
                chunk = y[start_sample:end_sample]

                if len(chunk) < sr_int * 0.5:
                    continue

                try:
                    tempo, _ = librosa.beat.beat_track(y=chunk, sr=sr_int)
                    segment.bpm = int(tempo)

                    chroma = librosa.feature.chroma(y=chunk, sr=sr_int)  # type: ignore
                    mean_chroma = np.mean(chroma, axis=1)

                    notes = [
                        "C",
                        "C#",
                        "D",
                        "D#",
                        "E",
                        "F",
                        "F#",
                        "G",
                        "G#",
                        "A",
                        "A#",
                        "B",
                    ]
                    key_idx = int(np.argmax(mean_chroma))
                    segment.mood = f"key center: {notes[key_idx]}"

                    rms = librosa.feature.rms(y=chunk)[0]
                    energy = np.mean(rms)

                    if energy < 0.02:
                        segment.valence = "low"
                        segment.arousal = "low"
                    elif energy < 0.05:
                        segment.valence = "medium"
                        segment.arousal = "medium"
                    else:
                        segment.valence = "high"
                        segment.arousal = "high"

                    segment.instrumentation = self._estimate_instruments(chunk, sr_int)

                except Exception:
                    continue

            return segments

        except Exception as e:
            print(f"    [MusicAnalyzer] Error analyzing features: {e}")
            return segments

    def _estimate_instruments(self, audio, sr: int) -> str:
        """Estimate dominant instruments using spectral features."""
        try:
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
            mean_centroid = np.mean(spectral_centroid)

            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            mfcc_means = np.mean(mfccs, axis=1)

            if mean_centroid < 1000:
                return "bass/low-end"
            elif mean_centroid < 2000:
                if mfcc_means[1] < -50:
                    return "piano"
                else:
                    return "strings"
            elif mean_centroid < 4000:
                return "guitar/synth"
            else:
                return "high-register"

        except Exception:
            return "unknown"
