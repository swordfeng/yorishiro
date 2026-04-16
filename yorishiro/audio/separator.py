"""Audio source separation: split voice and non-voice stems using Demucs.

Produces voice.flac (vocals) and nonvoice.flac (all other stems summed),
both at 44100 Hz stereo, extracted directly from the source video for
maximum quality.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

from yorishiro.audio import resample as audio_resample
from yorishiro.audio._speech_support import clear_torch_cache


@dataclass
class AudioSeparatorConfig:
    backend: str = "demucs"
    model: str = "htdemucs"
    device: str = "auto"
    sample_rate: int = 44100
    processing_chunk_seconds: float = 120.0
    processing_overlap_seconds: float = 1.0


class AudioSeparator:
    """Separate audio into voice and non-voice stems via Demucs."""

    def __init__(self, config: AudioSeparatorConfig | None = None) -> None:
        self.config = config or AudioSeparatorConfig()
        self._model = None

    def release_models(self) -> None:
        self._model = None
        clear_torch_cache()

    def separate(
        self,
        video_path: Path,
        output_dir: Path,
        force: bool = False,
    ) -> tuple[Path, Path]:
        """Separate voice and non-voice stems from a video file.

        Returns (voice_path, nonvoice_path).
        Writes voice.flac and nonvoice.flac to output_dir.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        voice_path = output_dir / "voice.flac"
        nonvoice_path = output_dir / "nonvoice.flac"
        voice_tmp_path = output_dir / "voice.flac.tmp"
        nonvoice_tmp_path = output_dir / "nonvoice.flac.tmp"

        if not force and voice_path.exists() and nonvoice_path.exists():
            return voice_path, nonvoice_path

        chunk_samples = max(
            int(self.config.processing_chunk_seconds * self.config.sample_rate),
            self.config.sample_rate,
        )
        overlap_samples = max(
            int(self.config.processing_overlap_seconds * self.config.sample_rate), 0
        )

        pending_voice = np.zeros((2, 0), dtype=np.float32)
        pending_nonvoice = np.zeros((2, 0), dtype=np.float32)
        processed_samples = 0
        total_seconds = self._audio_duration_seconds(video_path)
        total_minutes = (
            float(total_seconds / 60.0)
            if total_seconds is not None and total_seconds > 0
            else None
        )

        voice_tmp_path.unlink(missing_ok=True)
        nonvoice_tmp_path.unlink(missing_ok=True)
        try:
            with (
                sf.SoundFile(
                    str(voice_tmp_path),
                    mode="w",
                    samplerate=self.config.sample_rate,
                    channels=2,
                    subtype="PCM_16",
                    format="FLAC",
                ) as voice_file,
                sf.SoundFile(
                    str(nonvoice_tmp_path),
                    mode="w",
                    samplerate=self.config.sample_rate,
                    channels=2,
                    subtype="PCM_16",
                    format="FLAC",
                ) as nonvoice_file,
                tqdm(
                    total=total_minutes,
                    desc="  [AudioSeparator] Demucs",
                    unit="min",
                    bar_format="{l_bar}{bar}| {n:.1f}/{total_fmt} [{elapsed}<{remaining}]",
                ) as progress,
            ):
                print("  [AudioSeparator] Streaming separation in chunks ...")
                for audio_chunk in self._iter_audio_chunks(video_path, chunk_samples):
                    voice_chunk, nonvoice_chunk = self._run_demucs(audio_chunk)

                    write_voice, pending_voice = self._stitch_chunk(
                        pending_voice, voice_chunk, overlap_samples
                    )
                    write_nonvoice, pending_nonvoice = self._stitch_chunk(
                        pending_nonvoice, nonvoice_chunk, overlap_samples
                    )

                    if write_voice.shape[1] > 0:
                        voice_file.write(write_voice.T)
                    if write_nonvoice.shape[1] > 0:
                        nonvoice_file.write(write_nonvoice.T)

                    processed_samples += int(audio_chunk.shape[1])
                    chunk_minutes = (
                        audio_chunk.shape[1] / self.config.sample_rate / 60.0
                    )
                    progress.update(float(chunk_minutes))
                    progress.set_postfix_str(
                        f"processed={processed_samples / self.config.sample_rate / 60.0:.1f}m"
                    )

                if pending_voice.shape[1] > 0:
                    voice_file.write(pending_voice.T)
                if pending_nonvoice.shape[1] > 0:
                    nonvoice_file.write(pending_nonvoice.T)
        except Exception:
            voice_tmp_path.unlink(missing_ok=True)
            nonvoice_tmp_path.unlink(missing_ok=True)
            raise

        voice_tmp_path.replace(voice_path)
        nonvoice_tmp_path.replace(nonvoice_path)

        print(f"  [AudioSeparator] Wrote {voice_path.name} and {nonvoice_path.name}")
        return voice_path, nonvoice_path

    @staticmethod
    def _audio_duration_seconds(video_path: Path) -> float | None:
        import av

        container = av.open(str(video_path))
        try:
            audio_stream = next(
                (s for s in container.streams if s.type == "audio"), None
            )
            if (
                audio_stream is not None
                and audio_stream.duration is not None
                and audio_stream.time_base is not None
            ):
                return float(audio_stream.duration * audio_stream.time_base)
            if container.duration is not None:
                return float(container.duration) / 1_000_000.0
            return None
        finally:
            container.close()

    def _iter_audio_chunks(self, video_path: Path, chunk_samples: int):
        """Yield resampled stereo chunks as float32 arrays shaped (2, N).

        Audio is accumulated at the source sample rate, then each chunk is
        resampled to the target rate in one soxr call.  This avoids per-frame
        rounding drift that accumulates when resampling small codec frames
        individually (e.g. 48 kHz EAC3 1536-sample frames → 44.1 kHz loses
        ~0.2 samples/frame, adding up to >1 s over a feature-length film).
        """
        import av
        from av.audio.frame import AudioFrame

        input_container = av.open(str(video_path))
        audio_stream = next(
            (s for s in input_container.streams if s.type == "audio"), None
        )
        if audio_stream is None:
            raise ValueError(f"No audio stream found in {video_path}")

        # PyAV handles channel/layout normalisation only; sample-rate conversion
        # is deferred to soxr operating on whole chunks.
        resampler = av.AudioResampler(format="fltp", layout="stereo")
        buffered: deque[np.ndarray] = deque()
        buffered_samples = 0  # counted at *source* sample rate
        source_sr: int = 0
        emitted_any = False

        def _resample_chunk(chunk: np.ndarray) -> np.ndarray:
            """Resample a (2, N) array from source_sr to target SR."""
            if source_sr > 0 and source_sr != self.config.sample_rate:
                # soxr expects (samples, channels)
                return audio_resample.resample(
                    chunk.T,
                    orig_sr=source_sr,
                    target_sr=self.config.sample_rate,
                ).T.astype("float32", copy=False)
            return chunk

        def _source_chunk_samples() -> int:
            """How many source-rate samples correspond to one output chunk."""
            if source_sr > 0 and source_sr != self.config.sample_rate:
                return int(chunk_samples * source_sr / self.config.sample_rate)
            return chunk_samples

        try:
            for frame in input_container.decode(audio_stream):
                assert isinstance(frame, AudioFrame)
                for resampled in resampler.resample(frame):
                    arr = resampled.to_ndarray().astype("float32", copy=False)
                    if source_sr == 0:
                        source_sr = int(
                            getattr(resampled, "sample_rate", None)
                            or getattr(frame, "sample_rate", None)
                            or self.config.sample_rate
                        )
                    buffered.append(arr)
                    buffered_samples += int(arr.shape[1])

                    src_chunk = _source_chunk_samples()
                    while buffered_samples >= src_chunk:
                        out_parts: list[np.ndarray] = []
                        remaining = src_chunk
                        while remaining > 0:
                            part = buffered[0]
                            part_samples = int(part.shape[1])
                            if part_samples <= remaining:
                                out_parts.append(part)
                                buffered.popleft()
                                remaining -= part_samples
                            else:
                                out_parts.append(part[:, :remaining])
                                buffered[0] = part[:, remaining:]
                                remaining = 0
                        buffered_samples -= src_chunk
                        emitted_any = True
                        yield _resample_chunk(np.concatenate(out_parts, axis=1))
        finally:
            input_container.close()

        if buffered_samples <= 0 and not emitted_any:
            raise ValueError(f"No audio frames decoded from {video_path}")

        if buffered_samples > 0:
            yield _resample_chunk(np.concatenate(list(buffered), axis=1))

    @staticmethod
    def _stitch_chunk(
        pending_tail: np.ndarray, current: np.ndarray, overlap_samples: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Stitch one streamed chunk without rewriting time.

        Returns (write_block, next_pending_tail), both shaped (2, N).
        """
        # Outer separator chunks are adjacent slices of the source audio, not
        # overlapping windows. Crossfading them mixes different timestamps and
        # drops duration at each boundary, so streamed chunks must be appended
        # exactly as produced.
        if current.shape[1] == 0:
            return np.zeros((2, 0), dtype=np.float32), pending_tail

        if pending_tail.shape[1] == 0:
            return current, np.zeros((2, 0), dtype=np.float32)

        return (
            np.concatenate([pending_tail, current], axis=1),
            np.zeros((2, 0), dtype=np.float32),
        )

    def _run_demucs(self, audio):  # type: ignore[return]
        """Run Demucs on a (2, N) float32 array. Returns (voice, nonvoice) as (2, N) arrays."""
        import torch
        from yorishiro.utils import get_device

        try:
            from demucs import pretrained
            from demucs.apply import apply_model

            device_str = (
                get_device() if self.config.device == "auto" else self.config.device
            )
            device = torch.device(device_str)

            if self._model is None:
                self._model = pretrained.get_model(self.config.model)
                self._model.to(device)
                self._model.eval()

            # Create tensor on CPU and process per chunk to bound memory usage.
            tensor = torch.from_numpy(audio).unsqueeze(0)  # (1, 2, N) on CPU

            with torch.no_grad():
                sources = apply_model(
                    self._model,
                    tensor,
                    device=device,
                    # Keep Demucs internal splitting enabled so each streamed
                    # chunk can still exceed training window length safely.
                    split=True,
                    overlap=0.25,
                    progress=False,
                    num_workers=0,
                )  # type: ignore[call-arg]
                sources = sources[0].cpu().numpy()  # (stems, 2, N)

            vocals_idx = self._model.sources.index("vocals")
            voice = sources[vocals_idx]  # (2, N)
            nonvoice = sources.sum(axis=0) - voice  # (2, N)

            return voice, nonvoice

        except ImportError:
            print(
                "  [AudioSeparator] Demucs not installed — returning original audio as both stems"
            )
            return audio, audio
        except Exception as e:
            print(
                f"  [AudioSeparator] Demucs error: {e} — returning original audio as both stems"
            )
            return audio, audio
