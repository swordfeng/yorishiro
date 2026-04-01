"""Global speaker bank for cross-scene speaker consistency.

Manages speaker embeddings and maps local speaker IDs to global SPKR_XXX IDs.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from yorishiro.models.film_models import SpeakerBank


class SpeakerBankManagerConfig(BaseModel):
    embedding_backend: str = Field(default="pyannote", description="Embedding backend")
    similarity_threshold: float = Field(default=0.75, description="Threshold for speaker matching")


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
            import torch
            from pyannote.audio import Inference
            from pyannote.audio import Model

            if self._embedding_model is None:
                model = Model.from_pretrained(
                    "pyannote/embedding",
                    use_auth_token=False,
                )
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = model.to(device)
                self._embedding_model = Inference(model, window="whole")

            embedding = self._embedding_model(
                {"uri": audio_path.name, "audio": audio_path},
                start=start,
                end=end,
            )

            return embedding

        except Exception as e:
            print(f"    [SpeakerEmbedding] Error extracting embedding: {e}")
            return None

    def assign_global_speaker_id(
        self,
        local_speaker: str,
        embedding: np.ndarray | None,
        timestamp: float,
    ) -> str:
        """Assign a global speaker ID for a local speaker.

        Matches against existing speakers using embedding similarity.
        If no match or no embedding, creates a new global ID.
        """
        if embedding is not None:
            for spk_id, spk_emb in self._embeddings.items():
                similarity = self._cosine_similarity(embedding, spk_emb)
                if similarity >= self.config.similarity_threshold:
                    return spk_id

        global_id = f"SPKR_{len(self.speaker_bank.speakers) + 1:03d}"
        self.speaker_bank.add_speaker(global_id, timestamp)

        if embedding is not None:
            self._embeddings[global_id] = embedding

        return global_id

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