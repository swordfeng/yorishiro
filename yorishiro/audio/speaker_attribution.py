"""Speaker attribution runtime for `film.audio.speakers`."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from yorishiro.audio.speaker_bank import SpeakerBankManager, SpeakerBankManagerConfig
from yorishiro.models.film_models import SpeakerAttribution, SpeakerAttributionEntry, STTTranscript


@dataclass(frozen=True)
class SpeakerAttributorConfig:
    embedding_backend: str = "pyannote"
    similarity_threshold: float = 0.75
    embedding_min_duration_seconds: float = 0.5
    bank_enroll_min_duration_seconds: float = 1.0
    bank_enroll_min_confidence: float = -0.3
    hf_token_env: str = "YORISHIRO_HF_TOKEN"


class SpeakerAttributor:
    """Assign provisional speaker IDs to STT entries and build a speaker bank."""

    def __init__(self, config: SpeakerAttributorConfig | None = None) -> None:
        self.config = config or SpeakerAttributorConfig()

    def run(self, audio_path: Path, output_dir: Path) -> SpeakerAttribution:
        output_dir.mkdir(parents=True, exist_ok=True)
        stt = STTTranscript(**json.loads((output_dir / "stt.json").read_text(encoding="utf-8")))
        attribution = self._attribute(audio_path, stt, output_dir)
        out = output_dir / "speaker_attribution.json"
        out.write_text(attribution.model_dump_json(indent=2), encoding="utf-8")
        print(f"  [Speakers] Done — {len(attribution.entries)} entry attribution(s)")
        return attribution

    def _attribute(self, audio_path: Path, stt: STTTranscript, output_dir: Path) -> SpeakerAttribution:
        bank = SpeakerBankManager(
            SpeakerBankManagerConfig(
                embedding_backend=self.config.embedding_backend,
                similarity_threshold=self.config.similarity_threshold,
                hf_token_env=self.config.hf_token_env,
            )
        )
        next_speaker_num = 1
        cluster_counts: dict[str, int] = {}
        attribution_entries: list[SpeakerAttributionEntry] = []

        for idx, entry in enumerate(stt.entries):
            entry_id = f"utt_{idx:06d}"
            duration = max(entry.end - entry.start, 0.0)
            embedding = None
            if duration >= self.config.embedding_min_duration_seconds:
                embedding = bank.extract_speaker_embedding(audio_path, entry.start, entry.end)

            if embedding is None:
                attribution_entries.append(
                    SpeakerAttributionEntry(
                        entry_id=entry_id,
                        start=entry.start,
                        end=entry.end,
                        speaker_id="UNKNOWN",
                        similarity=None,
                        embedding_present=False,
                        enrolled=False,
                    )
                )
                continue

            speaker_id, similarity = self._assign_speaker_id(bank, embedding, next_speaker_num)
            if speaker_id == f"SPKR_{next_speaker_num:03d}":
                next_speaker_num += 1
                cluster_counts[speaker_id] = 0
            if bank.speaker_bank.get_speaker(speaker_id) is None:
                bank.speaker_bank.add_speaker(speaker_id, entry.start)

            cluster_counts[speaker_id] = cluster_counts.get(speaker_id, 0) + 1
            self._update_cluster_embedding(bank, speaker_id, embedding, cluster_counts[speaker_id])

            enrolled = (
                duration >= self.config.bank_enroll_min_duration_seconds
                and entry.confidence >= self.config.bank_enroll_min_confidence
            )
            if enrolled:
                info = bank.speaker_bank.get_speaker(speaker_id)
                if info is not None:
                    info.first_seen_time = min(info.first_seen_time, entry.start)

            attribution_entries.append(
                SpeakerAttributionEntry(
                    entry_id=entry_id,
                    start=entry.start,
                    end=entry.end,
                    speaker_id=speaker_id,
                    similarity=similarity,
                    embedding_present=True,
                    enrolled=enrolled,
                )
            )

        bank.save(output_dir)
        return SpeakerAttribution(entries=attribution_entries)

    def _assign_speaker_id(
        self,
        bank: SpeakerBankManager,
        embedding: np.ndarray,
        next_speaker_num: int,
    ) -> tuple[str, float | None]:
        best_speaker: str | None = None
        best_similarity: float | None = None
        for speaker_id, speaker_embedding in bank._embeddings.items():
            similarity = bank._cosine_similarity(embedding, speaker_embedding)
            if best_similarity is None or similarity > best_similarity:
                best_similarity = similarity
                best_speaker = speaker_id

        if best_speaker is not None and best_similarity is not None and best_similarity >= self.config.similarity_threshold:
            return best_speaker, best_similarity
        return f"SPKR_{next_speaker_num:03d}", best_similarity

    def _update_cluster_embedding(
        self,
        bank: SpeakerBankManager,
        speaker_id: str,
        embedding: np.ndarray,
        count: int,
    ) -> None:
        current = bank._embeddings.get(speaker_id)
        if current is None:
            bank._embeddings[speaker_id] = embedding
            return
        bank._embeddings[speaker_id] = ((current * (count - 1)) + embedding) / count
