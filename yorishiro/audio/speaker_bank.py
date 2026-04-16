"""Global speaker bank for cross-scene speaker consistency.

Manages speaker embeddings and maps local speaker IDs to global SPKR_XXX IDs.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import torch
from dataclasses import dataclass

from yorishiro.models.film_models import SpeakerBank
from yorishiro.utils import get_device


@dataclass
class SpeakerBankManagerConfig:
    # Which interface to use for speaker embedding extraction.
    # - "pyannote" (default): pyannote.audio Inference (supports pyannote/* and pyannote/wespeaker-* models)
    # - "wespeaker": native wespeaker SDK via ModelScope (for iic/* models, etc.)
    embedding_backend: str = "pyannote"
    # Model ID override. If None, a sensible default is used per backend:
    # - pyannote: pyannote/wespeaker-voxceleb-resnet34-LM
    # - wespeaker: iic/speech_eres2netv2_sv_zh-cn_16k-common
    embedding_model: str | None = None


class SpeakerBankManager:
    """Manages global speaker ID assignment across scenes."""

    def __init__(self, config: SpeakerBankManagerConfig | None = None):
        self.config = config or SpeakerBankManagerConfig()
        self._embedding_model = None
        self.speaker_bank = SpeakerBank()
        self._embeddings: dict[str, np.ndarray] = {}

    def release_models(self) -> None:
        from yorishiro.audio._speech_support import clear_torch_cache

        self._embedding_model = None
        clear_torch_cache()

    def save(self, output_dir: Path) -> None:
        """Save speaker bank to disk."""
        output_dir.mkdir(parents=True, exist_ok=True)
        bank_file = output_dir / "speaker_bank.json"
        embeddings_file = output_dir / "speaker_embeddings.pkl"

        bank_file.write_text(
            self.speaker_bank.model_dump_json(indent=2), encoding="utf-8"
        )

        with open(embeddings_file, "wb") as f:
            pickle.dump(self._embeddings, f)

    def load(self, output_dir: Path) -> bool:
        """Load speaker bank from disk. Returns True if loaded."""
        bank_file = output_dir / "speaker_bank.json"
        embeddings_file = output_dir / "speaker_embeddings.pkl"

        if not bank_file.exists():
            return False

        try:
            data = json.loads(bank_file.read_text(encoding="utf-8"))
            self.speaker_bank = SpeakerBank(**data)

            if embeddings_file.exists():
                with open(embeddings_file, "rb") as f:
                    self._embeddings = pickle.load(f)

            return True
        except Exception:
            return False

    def extract_speaker_embedding(
        self, audio_path: Path, start: float, end: float
    ) -> np.ndarray | None:
        """Extract speaker embedding for a segment."""
        try:
            import soundfile as sf

            backend = (self.config.embedding_backend or "pyannote").strip()

            if self._embedding_model is None:
                if backend == "wespeaker":
                    from wespeaker import Speaker  # type: ignore[import-untyped]  # ty:ignore[unresolved-import]

                    model_id = (
                        self.config.embedding_model
                        or "iic/speech_eres2netv2_sv_zh-cn_16k-common"
                    )
                    self._embedding_model = Speaker(model_id=model_id)
                else:
                    from pyannote.audio import Inference, Model

                    model_id = (
                        self.config.embedding_model
                        or "pyannote/wespeaker-voxceleb-resnet34-LM"
                    )
                    model = Model.from_pretrained(model_id)
                    device = torch.device(get_device())
                    model = model.to(device)  # type: ignore[union-attr]  # ty:ignore[unresolved-attribute]
                    self._embedding_model = Inference(model, window="whole")

            assert self._embedding_model is not None

            info = sf.info(str(audio_path))
            start_sample = int(start * info.samplerate)
            end_sample = int(end * info.samplerate)
            chunk, sr = sf.read(
                str(audio_path),
                start=start_sample,
                stop=end_sample,
                dtype="float32",
                always_2d=False,
            )

            if backend == "wespeaker":
                # wespeaker SDK expects a mono float32 numpy array
                audio_mono = chunk if chunk.ndim == 1 else chunk.mean(axis=1)
                embedding = self._embedding_model.extract_embedding_from_data(  # type: ignore[union-attr]  # ty:ignore[unresolved-attribute]
                    audio_mono, sr
                )
            else:
                t = torch.tensor(chunk)
                if t.ndim == 1:
                    waveform = t.unsqueeze(0)  # mono: (1, time)
                else:
                    waveform = t.T.contiguous()  # stereo: (channels, time)
                embedding = self._embedding_model(  # type: ignore[operator]
                    {"waveform": waveform, "sample_rate": int(sr)}
                )

            return embedding if isinstance(embedding, np.ndarray) else None

        except Exception as e:
            print(f"    [SpeakerEmbedding] Error extracting embedding: {e}")
            return None

    def confirm_speaker(
        self, speaker_id: str, character_name: str, scene_id: str | None = None
    ) -> None:
        """Confirm a speaker ID maps to a character name."""
        self.speaker_bank.confirm_speaker(speaker_id, character_name, scene_id)

    def get_speaker_name(self, speaker_id: str) -> str | None:
        """Get the character name for a speaker ID if confirmed."""
        return self.speaker_bank.speaker_map.get(speaker_id)

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        if a.shape != b.shape:
            return 0.0
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
