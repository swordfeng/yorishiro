"""Voice activity detection runtime for `film.audio.vad`."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VadConfig:
    vad_backend: str = "silero-vad"


class VadRunner:
    """Run voice activity detection and persist `vad.json`."""

    def __init__(self, config: VadConfig | None = None) -> None:
        self.config = config or VadConfig()

    def run(self, audio_path: Path, output_dir: Path) -> list[dict[str, float]]:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"  [VAD] Running on {audio_path.name} ...")
        segments = self._run_vad(audio_path)
        out = output_dir / "vad.json"
        out.write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  [VAD] Done — {len(segments)} speech segment(s)")
        return segments

    def _run_vad(self, audio_path: Path) -> list[dict[str, float]]:
        from silero_vad import get_speech_timestamps, load_silero_vad, read_audio

        model = load_silero_vad()
        wav = read_audio(str(audio_path))
        timestamps = get_speech_timestamps(wav, model, sampling_rate=16000, return_seconds=True)
        segments = [{"start": float(t["start"]), "end": float(t["end"])} for t in timestamps]
        return segments or [{"start": 0.0, "end": float("inf")}]
