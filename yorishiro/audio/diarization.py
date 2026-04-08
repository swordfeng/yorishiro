"""Speaker diarization runtime for `film.audio.diarize`."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import soundfile as sf
from tqdm import tqdm

from yorishiro.audio._speech_support import (
    DiarizationPipelineLike,
    get_diarization_pipeline,
    merge_chunk_speakers,
    save_speaker_bank,
    vad_chunk_boundaries,
)


@dataclass(frozen=True)
class DiarizerConfig:
    diarization_backend: str = "pyannote"
    diarization_model: str = "pyannote/speaker-diarization-3.1"
    diarization_batch_size: int = 32
    hf_token_env: str = "YORISHIRO_HF_TOKEN"


class Diarizer:
    """Run diarization and persist `diarization.json`."""

    def __init__(self, config: DiarizerConfig | None = None) -> None:
        self.config = config or DiarizerConfig()

    def run(self, audio_path: Path, output_dir: Path, force: bool = False) -> list[dict[str, Any]]:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"  [Diarization] Running on {audio_path.name} ...")
        turns = self._run_diarization(audio_path, output_dir=output_dir, force=force)
        out = output_dir / "diarization.json"
        out.write_text(json.dumps(turns, ensure_ascii=False, indent=2), encoding="utf-8")
        speakers = {cast(str, turn["speaker"]) for turn in turns}
        print(
            f"  [Diarization] Done — {len(turns)} turn(s), {len(speakers)} speaker(s): {', '.join(sorted(speakers))}"
        )
        return turns

    def _load_diarization_pipeline(self) -> DiarizationPipelineLike:
        return get_diarization_pipeline(self.config)

    def _diarize_chunk(
        self,
        pipeline: DiarizationPipelineLike,
        audio: np.ndarray,
        sample_rate: int,
        chunk_idx: int,
        chunk_start: float,
    ) -> tuple[list[dict[str, Any]], np.ndarray | None, list[str]]:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            tmp_path = Path(handle.name)
        try:
            sf.write(str(tmp_path), audio, sample_rate)
            with tqdm(
                total=1.0,
                desc=f"    [Diarization] chunk {chunk_idx}",
                unit="%",
                bar_format="{l_bar}{bar}| {elapsed}<{remaining}",
            ) as progress:
                last = 0.0

                def hook(
                    _step_name: str,
                    _step_artifact: object,
                    file: object = None,
                    total: int | None = None,
                    completed: int | None = None,
                ) -> None:
                    del file
                    nonlocal last
                    if completed is not None and total is not None and total > 0:
                        ratio = completed / total
                        progress.update(ratio - last)
                        last = ratio

                result = pipeline(str(tmp_path), hook=hook)

            ann = result.exclusive_speaker_diarization if hasattr(result, "exclusive_speaker_diarization") else result
            embeddings: np.ndarray | None = result.speaker_embeddings if hasattr(result, "speaker_embeddings") else None
            full_ann = result.speaker_diarization if hasattr(result, "speaker_diarization") else ann
            speakers_local = full_ann.labels() if hasattr(full_ann, "labels") else []

            turns: list[dict[str, Any]] = []
            for segment, _, speaker in ann.itertracks(yield_label=True):
                turns.append({
                    "speaker": speaker,
                    "start": round(chunk_start + segment.start, 3),
                    "end": round(chunk_start + segment.end, 3),
                })
            return turns, embeddings, speakers_local
        finally:
            tmp_path.unlink(missing_ok=True)

    def _run_diarization(
        self,
        audio_path: Path,
        output_dir: Path | None = None,
        force: bool = False,
    ) -> list[dict[str, Any]]:
        try:
            hf_token = os.environ.get(self.config.hf_token_env)
            if not hf_token:
                print(f"    [Diarization] {self.config.hf_token_env} not set, using single-speaker fallback")
                return [{"speaker": "SPEAKER_00", "start": 0.0, "end": float("inf")}]

            info = sf.info(str(audio_path))
            total_duration = info.duration
            sample_rate = info.samplerate
            source_mtime = audio_path.stat().st_mtime

            vad_segments: list[dict[str, float]] = []
            if output_dir and (output_dir / "vad.json").exists():
                vad_segments = json.loads((output_dir / "vad.json").read_text(encoding="utf-8"))

            boundaries = vad_chunk_boundaries(total_duration, vad_segments)
            num_chunks = len(boundaries) - 1
            checkpoint_dir = output_dir / ".diarization_checkpoints" if output_dir else None

            if force:
                import shutil

                if checkpoint_dir and checkpoint_dir.exists():
                    shutil.rmtree(checkpoint_dir)
                    print("    [Diarization] Cleared checkpoints (force)")
                if output_dir:
                    out_json = output_dir / "diarization.json"
                    if out_json.exists():
                        out_json.unlink()
                        print("    [Diarization] Cleared output (force)")

            if checkpoint_dir:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)

            pipeline = self._load_diarization_pipeline()

            chunk_results: list[dict[str, Any]] = []
            for chunk_idx in range(num_chunks):
                chunk_start = boundaries[chunk_idx]
                chunk_end = boundaries[chunk_idx + 1]
                checkpoint_file = checkpoint_dir / f"chunk_{chunk_idx:04d}.json" if checkpoint_dir else None

                if checkpoint_file and checkpoint_file.exists() and not force:
                    checkpoint = json.loads(checkpoint_file.read_text(encoding="utf-8"))
                    if checkpoint.get("source_mtime") == source_mtime:
                        print(f"    [Diarization] chunk {chunk_idx} — resuming from checkpoint")
                        npy_file = checkpoint_file.with_suffix(".npy")
                        if npy_file.exists():
                            checkpoint["embeddings"] = np.load(str(npy_file))
                        else:
                            checkpoint["embeddings"] = None
                        chunk_results.append(checkpoint)
                        continue
                    print(f"    [Diarization] chunk {chunk_idx} — checkpoint stale, reprocessing")

                start_sample = int(chunk_start * sample_rate)
                end_sample = int(chunk_end * sample_rate)
                chunk_audio, _ = sf.read(str(audio_path), start=start_sample, stop=end_sample, dtype="float32")

                turns, embeddings, speakers_local = self._diarize_chunk(
                    pipeline,
                    chunk_audio,
                    sample_rate,
                    chunk_idx,
                    chunk_start,
                )

                checkpoint_data: dict[str, Any] = {
                    "chunk_idx": chunk_idx,
                    "source_mtime": source_mtime,
                    "turns": turns,
                    "speakers_local": speakers_local,
                    "embeddings": embeddings,
                }
                if checkpoint_file:
                    checkpoint_json = {key: value for key, value in checkpoint_data.items() if key != "embeddings"}
                    checkpoint_file.write_text(
                        json.dumps(checkpoint_json, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    if embeddings is not None:
                        np.save(str(checkpoint_file.with_suffix(".npy")), embeddings)

                chunk_results.append(checkpoint_data)

            if num_chunks == 1:
                all_turns: list[dict[str, Any]] = []
                sorted_speakers = sorted({cast(str, turn["speaker"]) for turn in cast(list[dict[str, Any]], chunk_results[0]["turns"])})
                mapping = {speaker: f"SPEAKER_{idx:02d}" for idx, speaker in enumerate(sorted_speakers)}
                for turn in cast(list[dict[str, Any]], chunk_results[0]["turns"]):
                    all_turns.append({**turn, "speaker": mapping.get(cast(str, turn["speaker"]), cast(str, turn["speaker"]))})
                segments = all_turns

                speaker_embeddings: dict[str, np.ndarray] = {}
                embeddings = cast(np.ndarray | None, chunk_results[0].get("embeddings"))
                speakers_local = cast(list[str], chunk_results[0].get("speakers_local", []))
                if embeddings is not None:
                    for idx, local_id in enumerate(speakers_local):
                        speaker_embeddings[mapping.get(local_id, local_id)] = embeddings[idx]
            else:
                segments, speaker_embeddings = merge_chunk_speakers(chunk_results)

            if output_dir and segments:
                save_speaker_bank(output_dir, segments, speaker_embeddings)

            return segments if segments else [{"speaker": "SPEAKER_00", "start": 0.0, "end": float("inf")}]
        except Exception as exc:
            print(f"    [Diarization] Error: {exc}, using single-speaker fallback")
            return [{"speaker": "SPEAKER_00", "start": 0.0, "end": float("inf")}]
