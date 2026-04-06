"""Audio source separation: split voice and non-voice stems using Demucs.

Produces voice.flac (vocals) and nonvoice.flac (all other stems summed),
both at 44100 Hz stereo, extracted directly from the source video for
maximum quality.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import soundfile as sf


@dataclass
class AudioSeparatorConfig:
    backend: str = "demucs"
    model: str = "htdemucs"
    device: str = "auto"
    sample_rate: int = 44100


class AudioSeparator:
    """Separate audio into voice and non-voice stems via Demucs."""

    def __init__(self, config: AudioSeparatorConfig | None = None) -> None:
        self.config = config or AudioSeparatorConfig()
        self._model = None

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

        if not force and voice_path.exists() and nonvoice_path.exists():
            return voice_path, nonvoice_path

        audio = self._extract_audio(video_path)
        voice, nonvoice = self._run_demucs(audio)

        sf.write(str(voice_path), voice.T, self.config.sample_rate)
        sf.write(str(nonvoice_path), nonvoice.T, self.config.sample_rate)

        print(f"  [AudioSeparator] Wrote {voice_path.name} and {nonvoice_path.name}")
        return voice_path, nonvoice_path

    def _extract_audio(self, video_path: Path):  # type: ignore[return]
        """Extract 44kHz stereo audio from video as a numpy array."""
        import av
        import numpy as np
        from av.audio.frame import AudioFrame

        input_container = av.open(str(video_path))
        audio_stream = next(
            (s for s in input_container.streams if s.type == "audio"), None
        )
        if audio_stream is None:
            raise ValueError(f"No audio stream found in {video_path}")

        resampler = av.AudioResampler(format="fltp", layout="stereo", rate=self.config.sample_rate)
        chunks = []
        for frame in input_container.decode(audio_stream):
            assert isinstance(frame, AudioFrame)
            for resampled in resampler.resample(frame):
                arr = resampled.to_ndarray()  # (channels, samples)
                chunks.append(arr)

        input_container.close()

        if not chunks:
            raise ValueError(f"No audio frames decoded from {video_path}")

        audio = np.concatenate(chunks, axis=1).astype("float32")  # (2, N)
        return audio

    def _run_demucs(self, audio):  # type: ignore[return]
        """Run Demucs on a (2, N) float32 array. Returns (voice, nonvoice) as (2, N) arrays."""
        import torch
        from yorishiro.utils import get_device

        try:
            from demucs import pretrained
            from demucs.apply import apply_model

            device_str = get_device() if self.config.device == "auto" else self.config.device
            device = torch.device(device_str)

            if self._model is None:
                self._model = pretrained.get_model(self.config.model)
                self._model.to(device)
                self._model.eval()

            # Create tensor on CPU, let apply_model handle GPU transfer with chunking
            tensor = torch.from_numpy(audio).unsqueeze(0)  # (1, 2, N) on CPU

            print(f"  [AudioSeparator] Processing {(audio.shape[1] / self.config.sample_rate / 60):.1f} min of audio...")
            with torch.no_grad():
                sources = apply_model(
                    self._model, tensor, device=device,
                    split=True, progress=True,
                )  # type: ignore[call-arg]
                sources = sources[0].cpu().numpy()  # (stems, 2, N)

            vocals_idx = self._model.sources.index("vocals")
            voice = sources[vocals_idx]  # (2, N)
            nonvoice = sources.sum(axis=0) - voice  # (2, N)

            return voice, nonvoice

        except ImportError:
            print("  [AudioSeparator] Demucs not installed — returning original audio as both stems")
            return audio, audio
        except Exception as e:
            print(f"  [AudioSeparator] Demucs error: {e} — returning original audio as both stems")
            return audio, audio
