"""Global speaker bank for cross-scene speaker consistency.

Manages speaker embeddings and maps local speaker IDs to global SPKR_XXX IDs.
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import numpy as np
import torch
from dataclasses import dataclass

from yorishiro.models.film_models import SpeakerBank
from yorishiro.utils import get_device


@dataclass
class SpeakerBankManagerConfig:
    # Which speaker embedding model to use.
    # Supported values:
    # - "wespeaker" (default): pyannote/wespeaker-voxceleb-resnet34-LM
    # - "pyannote": pyannote/embedding
    # - any Hugging Face model id (e.g. "eek/wespeaker-voxceleb-resnet293-LM")
    embedding_backend: str = "wespeaker"
    hf_token_env: str = "YORISHIRO_HF_TOKEN"


class SpeakerBankManager:
    """Manages global speaker ID assignment across scenes."""

    def __init__(self, config: SpeakerBankManagerConfig | None = None):
        self.config = config or SpeakerBankManagerConfig()
        self._embedding_model = None
        self.speaker_bank = SpeakerBank()
        self._embeddings: dict[str, np.ndarray] = {}

    def save(self, output_dir: Path) -> None:
        """Save speaker bank to disk."""
        output_dir.mkdir(parents=True, exist_ok=True)
        bank_file = output_dir / "speaker_bank.json"
        embeddings_file = output_dir / "speaker_embeddings.pkl"

        bank_file.write_text(self.speaker_bank.model_dump_json(indent=2), encoding="utf-8")

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

    def extract_speaker_embedding(self, audio_path: Path, start: float, end: float) -> np.ndarray | None:
        """Extract speaker embedding for a segment."""
        try:
            from pyannote.audio import Inference
            from pyannote.audio import Model

            if self._embedding_model is None:
                hf_token = os.environ.get(self.config.hf_token_env)
                if not hf_token:
                    print(f"    [SpeakerEmbedding] {self.config.hf_token_env} not set, skipping embedding extraction")
                    return None

                backend = (self.config.embedding_backend or "").strip()
                if backend in {"wespeaker", "wespeaker_resnet34", "pyannote/wespeaker-voxceleb-resnet34-LM"}:
                    model_id = "pyannote/wespeaker-voxceleb-resnet34-LM"
                elif backend in {"pyannote", "pyannote/embedding"}:
                    model_id = "pyannote/embedding"
                elif "/" in backend:
                    model_id = backend
                else:
                    model_id = "pyannote/wespeaker-voxceleb-resnet34-LM"

                model = Model.from_pretrained(model_id, token=hf_token)
                device = torch.device(get_device())
                model = model.to(device)  # type: ignore[union-attr]  # ty:ignore[unresolved-attribute]
                self._embedding_model = Inference(model, window="whole")

            assert self._embedding_model is not None
            import soundfile as sf
            info = sf.info(str(audio_path))
            start_sample = int(start * info.samplerate)
            end_sample = int(end * info.samplerate)
            chunk, sr = sf.read(str(audio_path), start=start_sample, stop=end_sample, dtype="float32", always_2d=False)
            t = torch.tensor(chunk)
            if t.ndim == 1:
                waveform = t.unsqueeze(0)        # mono: (1, time)
            else:
                waveform = t.T.contiguous()      # stereo: (channels, time)
            embedding = self._embedding_model({"waveform": waveform, "sample_rate": int(sr)})

            return embedding if isinstance(embedding, np.ndarray) else None

        except Exception as e:
            print(f"    [SpeakerEmbedding] Error extracting embedding: {e}")
            return None

    def confirm_speaker(self, speaker_id: str, character_name: str, scene_id: str | None = None) -> None:
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
