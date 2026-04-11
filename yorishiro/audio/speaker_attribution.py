"""Speaker attribution runtime for `film.audio.speakers`.

Uses windowed embedding extraction with per-segment voting:
1. Split each STT segment into short overlapping windows (1.5s, 0.75s hop)
2. Extract speaker embedding per window
3. Cluster embeddings globally (default: utterance-averaged UMAP+HDBSCAN)
4. Assign each STT segment a speaker via weighted majority vote
5. Short segments with no windows fall back to temporal proximity
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any, TypedDict, cast

import numpy as np
import soundfile as sf
from scipy.spatial.distance import pdist
from tqdm import tqdm

from yorishiro.audio.speaker_bank import SpeakerBankManager, SpeakerBankManagerConfig
from yorishiro.models.film_models import (
    SpeakerAttribution,
    SpeakerAttributionEntry,
    SpeakerClusterQuality,
    STTTranscript,
    WindowVote,
)


@dataclass(frozen=True)
class EmbeddingWindow:
    stt_idx: int
    start: float
    end: float


@dataclass(frozen=True)
class ClusteringScoreBreakdown:
    score: float
    silhouette: float
    nn_purity: float
    disagreement_rate: float
    voted_separation: float
    singleton_point_fraction: float
    noise_fraction: float
    n_clusters: int
    max_cluster_ratio: float
    mean_persistence: float


class SweepCandidateRow(TypedDict):
    nn: int
    dim: int
    mcs: int
    score: float
    clusters: int
    n_noise: int
    top_str: str
    labels: np.ndarray
    nn_purity: float
    disagree: float
    sep: float
    silhouette: float
    singleton: float
    noise: float
    coarse: float
    persistence: float


@dataclass(frozen=True)
class SpeakerAttributorConfig:
    embedding_backend: str = "wespeaker"
    similarity_threshold: float = 0.75
    diagnostics_enabled: bool = False
    clustering_method: str = "umap_hdbscan_auto"
    window_duration: float = 1.5
    window_hop: float = 0.75
    min_window_duration: float = 0.8
    energy_threshold_db: float = -40.0
    norm_filter_sigma: float = 5.0
    coherence_threshold: float = 0.3
    min_utterance_duration_cluster: float = 1.5
    min_utterance_coherence_cluster: float = 0.4
    # If a window's best centroid cosine similarity is below this threshold,
    # the window contributes no votes.
    min_vote_similarity: float = 0.2
    hdbscan_min_cluster_size: int = 10
    umap_n_neighbors: int = 5
    umap_min_dist: float = 0.0
    umap_n_components: int = 5
    utterance_aggregation: str = "medoid"
    hf_token_env: str = "YORISHIRO_HF_TOKEN"


class SpeakerAttributor:
    """Assign provisional speaker IDs to STT entries via windowed embedding + voting."""

    def __init__(self, config: SpeakerAttributorConfig | None = None) -> None:
        self.config = config or SpeakerAttributorConfig()

    def run(
        self, audio_path: Path, output_dir: Path, force: bool = False
    ) -> SpeakerAttribution:
        output_dir.mkdir(parents=True, exist_ok=True)
        stt = STTTranscript(
            **json.loads((output_dir / "stt.json").read_text(encoding="utf-8"))
        )
        attribution = self._attribute(audio_path, stt, output_dir, force=force)
        out = output_dir / "speaker_attribution.json"
        out.write_text(attribution.model_dump_json(indent=2), encoding="utf-8")
        num_speakers = len(
            {e.speaker_id for e in attribution.entries if e.speaker_id != "UNKNOWN"}
        )
        print(
            f"  [Speakers] Done — {len(attribution.entries)} entry attribution(s), {num_speakers} speaker(s)"
        )
        return attribution

    def _attribute(
        self,
        audio_path: Path,
        stt: STTTranscript,
        output_dir: Path,
        force: bool = False,
    ) -> SpeakerAttribution:
        bank = SpeakerBankManager(
            SpeakerBankManagerConfig(
                embedding_backend=self.config.embedding_backend,
                hf_token_env=self.config.hf_token_env,
            )
        )

        # ── Phase 1: Extract windows ────────────────────────────────
        windows = self._extract_windows(stt)
        print(
            f"  [Speakers] {len(windows)} window(s) from {len(stt.entries)} segment(s)"
        )

        if not windows:
            print("  [Speakers] No windows — assigning all entries as UNKNOWN")
            return self._unknown_attribution(stt, output_dir, bank)

        # ── Phase 2: Embed windows ───────────────────────────────────
        valid_indices_all, embeddings_all_raw = self._embed_windows(
            windows, audio_path, bank, output_dir, stt=stt, force=force
        )

        print(
            f"  [Speakers] {len(embeddings_all_raw)} valid embedding(s) from {len(windows)} window(s)"
        )

        if not embeddings_all_raw:
            print("  [Speakers] No valid embeddings — assigning all entries as UNKNOWN")
            return self._unknown_attribution(stt, output_dir, bank)

        X_all_raw = np.stack(embeddings_all_raw)

        # ── Phase 2b-d: Filters for clustering only ─────────────────
        # We still assign/vote using *all* embedded windows (pre-filter).
        # Filters only decide which windows are trusted to form clusters/centroids.

        valid_indices_cluster, embeddings_cluster_raw = (
            list(valid_indices_all),
            list(embeddings_all_raw),
        )

        # Exclude short utterances from clustering (they still get voted later).
        valid_indices_cluster, embeddings_cluster_raw = (
            self._filter_by_utterance_duration(
                valid_indices_cluster, embeddings_cluster_raw, windows, stt
            )
        )

        # Per-utterance MAD norm filter (on raw norms).
        valid_indices_cluster, embeddings_cluster_raw = self._filter_by_utterance_norm(
            valid_indices_cluster, embeddings_cluster_raw, windows, stt
        )

        if embeddings_cluster_raw:
            Xc_raw = np.stack(embeddings_cluster_raw)
            norms_c = np.linalg.norm(Xc_raw, axis=1, keepdims=True)
            norms_c = np.maximum(norms_c, 1e-10)
            Xc = Xc_raw / norms_c
            embeddings_cluster = [Xc[i] for i in range(len(valid_indices_cluster))]
        else:
            embeddings_cluster = []

        # L2-normalize *all* embeddings for centroid assignment/voting.
        norms_all = np.linalg.norm(X_all_raw, axis=1, keepdims=True)
        norms_all = np.maximum(norms_all, 1e-10)
        X_all = X_all_raw / norms_all
        embeddings_all = [X_all[i] for i in range(len(valid_indices_all))]

        # In-utterance coherence filter (on normalized embeddings).
        valid_indices_cluster, embeddings_cluster = self._filter_by_utterance_coherence(
            valid_indices_cluster, embeddings_cluster, windows, stt
        )

        # Exclude low-coherence utterances entirely from clustering.
        valid_indices_cluster, embeddings_cluster = (
            self._filter_by_utterance_low_coherence(
                valid_indices_cluster, embeddings_cluster, windows, stt
            )
        )

        if len(valid_indices_cluster) == 0:
            print(
                "  [Speakers] No clusterable embeddings after filtering — assigning by temporal proximity only"
            )
            attribution_entries = self._vote_speakers(
                stt=stt,
                windows=windows,
                valid_indices_all=valid_indices_all,
                embeddings_all=embeddings_all,
                speaker_centroids={},
            )
            bank.save(output_dir)
            return SpeakerAttribution(entries=attribution_entries)

        X = np.stack(embeddings_cluster)

        # ── Phase 3: Cluster ─────────────────────────────────────────
        if len(valid_indices_cluster) == 1:
            cluster_labels = np.array([1])
        else:
            if self.config.diagnostics_enabled:
                self._print_diagnostics(X, windows, valid_indices_cluster)
            cluster_labels = self._cluster_windows(X, windows, valid_indices_cluster)

        unique_labels_in_cluster = set(int(x) for x in cluster_labels if x >= 0)
        label_to_speaker: dict[int, str] = {}
        for idx, label in enumerate(sorted(unique_labels_in_cluster)):
            label_to_speaker[label] = f"SPKR_{idx + 1:03d}"

        print(
            f"  [Speakers] Clustered {len(embeddings_cluster)} window(s) into {len(unique_labels_in_cluster)} speaker(s)"
        )

        # ── Compute speaker centroids from window embeddings ────────
        speaker_emb_accum: dict[str, list[np.ndarray]] = defaultdict(list)
        for pos in range(len(valid_indices_cluster)):
            label = int(cluster_labels[pos])
            if label < 0:
                continue
            speaker_id = label_to_speaker[label]
            speaker_emb_accum[speaker_id].append(embeddings_cluster[pos])

        speaker_centroids_raw: dict[str, np.ndarray] = {
            speaker_id: np.mean(np.stack(embs), axis=0)
            for speaker_id, embs in speaker_emb_accum.items()
        }
        # Normalize centroids for cosine similarity comparisons.
        speaker_centroids: dict[str, np.ndarray] = {}
        for speaker_id, c in speaker_centroids_raw.items():
            n = float(np.linalg.norm(c))
            speaker_centroids[speaker_id] = c / max(n, 1e-10)

        # ── Phase 4: Vote ────────────────────────────────────────────
        attribution_entries = self._vote_speakers(
            stt=stt,
            windows=windows,
            valid_indices_all=valid_indices_all,
            embeddings_all=embeddings_all,
            speaker_centroids=speaker_centroids,
        )

        # ── Phase 5: Save speaker bank ───────────────────────────────
        speaker_first_time: dict[str, float] = {}
        for idx, entry in enumerate(stt.entries):
            sid = attribution_entries[idx].speaker_id
            if sid != "UNKNOWN" and (
                sid not in speaker_first_time or entry.start < speaker_first_time[sid]
            ):
                speaker_first_time[sid] = entry.start

        for speaker_id in sorted(speaker_centroids.keys()):
            bank.speaker_bank.add_speaker(
                speaker_id, speaker_first_time.get(speaker_id, 0.0)
            )
            bank._embeddings[speaker_id] = speaker_centroids[speaker_id]

        bank.speaker_bank.cluster_quality = self._compute_cluster_quality(
            attribution_entries=attribution_entries,
            speaker_centroids=speaker_centroids,
            cluster_labels=cluster_labels,
            label_to_speaker=label_to_speaker,
            X_cluster=X,
            valid_indices_cluster=valid_indices_cluster,
            windows=windows,
        )

        for cq in bank.speaker_bank.cluster_quality:
            nearest = (
                f"nearest={cq.nearest_speaker}({cq.nearest_speaker_similarity:.3f})"
                if cq.nearest_speaker
                else "nearest=N/A"
            )
            intra = (
                f"intra={cq.intra_cluster_similarity:.3f}"
                if cq.intra_cluster_similarity is not None
                else "intra=N/A"
            )
            mean_s = (
                f"mean_sim={cq.mean_similarity:.3f}"
                if cq.mean_similarity is not None
                else "mean_sim=N/A"
            )
            print(
                f"    {cq.speaker_id}: {cq.num_utterances}utt/{cq.num_enrolled}enrolled/{cq.num_windows}win, "
                f"{mean_s}, {intra}, {nearest}"
            )

        bank.save(output_dir)
        return SpeakerAttribution(entries=attribution_entries)

    def _extract_windows(self, stt: STTTranscript) -> list[EmbeddingWindow]:
        windows: list[EmbeddingWindow] = []
        window_dur = self.config.window_duration
        window_hop = self.config.window_hop
        min_dur = self.config.min_window_duration

        for idx, entry in enumerate(stt.entries):
            duration = entry.end - entry.start
            if duration < 0.3:
                continue
            if duration < min_dur:
                # Still embed short utterances (single window) so they can be attributed
                # unless similarity to known speakers is very low.
                windows.append(
                    EmbeddingWindow(stt_idx=idx, start=entry.start, end=entry.end)
                )
                continue
            if duration < window_dur:
                windows.append(
                    EmbeddingWindow(stt_idx=idx, start=entry.start, end=entry.end)
                )
                continue

            t = entry.start
            while t + window_dur <= entry.end + 1e-6:
                win_end = min(t + window_dur, entry.end)
                windows.append(EmbeddingWindow(stt_idx=idx, start=t, end=win_end))
                t += window_hop

            tail_start = entry.end - window_dur
            if tail_start > t - window_hop + 1e-6 and tail_start >= entry.start:
                windows.append(
                    EmbeddingWindow(stt_idx=idx, start=tail_start, end=entry.end)
                )

        return windows

    def _window_has_energy(self, audio_path: Path, win: EmbeddingWindow) -> bool:
        try:
            info = sf.info(str(audio_path))
            start_sample = int(win.start * info.samplerate)
            end_sample = int(win.end * info.samplerate)
            chunk, _ = sf.read(
                str(audio_path),
                start=start_sample,
                stop=end_sample,
                dtype="float32",
                always_2d=False,
            )
            if chunk.ndim > 1:
                chunk = chunk.mean(axis=1)
            rms = np.sqrt(np.mean(chunk**2))
            if rms < 1e-10:
                return False
            rms_db = 20.0 * np.log10(rms)
            return rms_db >= self.config.energy_threshold_db
        except Exception:
            return False

    def _embed_windows(
        self,
        windows: list[EmbeddingWindow],
        audio_path: Path,
        bank: SpeakerBankManager,
        output_dir: Path,
        stt: STTTranscript | None = None,
        force: bool = False,
    ) -> tuple[list[int], list[np.ndarray]]:
        valid_indices: list[int] = []
        embeddings: list[np.ndarray] = []
        cache = self._load_embedding_cache(
            audio_path, output_dir, stt=stt, force=force
        )

        print(f"  [Speakers] Extracting embeddings for {len(windows)} window(s) ...")
        with tqdm(
            total=len(windows),
            desc="    [Speakers] Embed",
            unit="win",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        ) as progress:
            for i, win in enumerate(windows):
                if not self._window_has_energy(audio_path, win):
                    progress.update(1)
                    continue

                cache_key = self._embedding_cache_key(win)
                emb = cache.get(cache_key)
                if emb is None:
                    emb = bank.extract_speaker_embedding(audio_path, win.start, win.end)
                    if emb is not None:
                        cache[cache_key] = np.asarray(emb)
                if emb is not None:
                    valid_indices.append(i)
                    embeddings.append(emb)
                progress.update(1)

        self._save_embedding_cache(audio_path, output_dir, cache, stt=stt)
        return valid_indices, embeddings

    @staticmethod
    def _embedding_cache_path(output_dir: Path) -> Path:
        return output_dir / "speaker_embedding_cache.npz"

    def _embedding_cache_key(self, win: EmbeddingWindow) -> str:
        return f"{self.config.embedding_backend}|{win.stt_idx}|{win.start:.6f}|{win.end:.6f}"

    def _embedding_cache_source_sig(self, audio_path: Path) -> str:
        stat = audio_path.stat()
        raw = f"{audio_path.resolve()}|{stat.st_mtime_ns}|{stat.st_size}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _embedding_cache_stt_sig(stt: STTTranscript | None) -> str:
        if stt is None:
            return "none"
        stt_payload = stt.model_dump(mode="python")
        raw = json.dumps(stt_payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _embedding_cache_context_sig(
        self, audio_path: Path, stt: STTTranscript | None
    ) -> str:
        cfg = asdict(self.config)
        extraction_cfg = {
            "embedding_backend": cfg["embedding_backend"],
            "window_duration": cfg["window_duration"],
            "window_hop": cfg["window_hop"],
            "min_window_duration": cfg["min_window_duration"],
            "energy_threshold_db": cfg["energy_threshold_db"],
        }
        payload = {
            "schema_version": 2,
            "audio_sig": self._embedding_cache_source_sig(audio_path),
            "stt_sig": self._embedding_cache_stt_sig(stt),
            "extraction_cfg": extraction_cfg,
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _load_embedding_cache(
        self,
        audio_path: Path,
        output_dir: Path,
        stt: STTTranscript | None = None,
        force: bool = False,
    ) -> dict[str, np.ndarray]:
        if force:
            return {}
        cache_path = self._embedding_cache_path(output_dir)
        if not cache_path.exists():
            return {}
        try:
            data = np.load(cache_path, allow_pickle=True)
            schema_version = int(data["schema_version"].item())
            context_sig = str(data["context_sig"].item())
            if schema_version != 2:
                return {}
            if context_sig != self._embedding_cache_context_sig(audio_path, stt):
                return {}
            keys = data["keys"].tolist()
            values = data["values"]
            return {str(k): values[i] for i, k in enumerate(keys)}
        except Exception:
            return {}

    def _save_embedding_cache(
        self,
        audio_path: Path,
        output_dir: Path,
        cache: dict[str, np.ndarray],
        stt: STTTranscript | None = None,
    ) -> None:
        if not cache:
            return
        cache_path = self._embedding_cache_path(output_dir)
        keys = sorted(cache.keys())
        values = np.stack([np.asarray(cache[k], dtype=np.float32) for k in keys])
        np.savez(
            cache_path,
            schema_version=np.array(2, dtype=np.int64),
            context_sig=np.array(
                self._embedding_cache_context_sig(audio_path, stt), dtype=object
            ),
            keys=np.array(keys, dtype=object),
            values=values,
        )

    def _filter_by_utterance_duration(
        self,
        valid_indices: list[int],
        embeddings: list[np.ndarray],
        windows: list[EmbeddingWindow],
        stt: STTTranscript,
    ) -> tuple[list[int], list[np.ndarray]]:
        if not embeddings or self.config.min_utterance_duration_cluster <= 0:
            return valid_indices, embeddings
        min_dur = self.config.min_utterance_duration_cluster
        seg_windows: dict[int, list[int]] = defaultdict(list)
        for pos, win_idx in enumerate(valid_indices):
            seg_windows[windows[win_idx].stt_idx].append(pos)

        keep = set(range(len(valid_indices)))
        filtered_segs = 0
        for stt_idx, positions in seg_windows.items():
            entry = stt.entries[stt_idx]
            duration = entry.end - entry.start
            if duration < min_dur:
                for pos in positions:
                    keep.discard(pos)
                filtered_segs += 1

        if filtered_segs > 0:
            total = len(valid_indices) - len(keep)
            print(
                f"  [Speakers] Duration filter: discarded {total} window(s) from "
                f"{filtered_segs} short utterance(s) (min_dur={min_dur}s)"
            )

        kept_indices = [valid_indices[i] for i in sorted(keep)]
        kept_embeddings = [embeddings[i] for i in sorted(keep)]
        return kept_indices, kept_embeddings

    def _filter_by_utterance_norm(
        self,
        valid_indices: list[int],
        embeddings: list[np.ndarray],
        windows: list[EmbeddingWindow],
        stt: STTTranscript,
    ) -> tuple[list[int], list[np.ndarray]]:
        if not embeddings or self.config.norm_filter_sigma <= 0:
            return valid_indices, embeddings
        seg_windows: dict[int, list[int]] = defaultdict(list)
        for pos, win_idx in enumerate(valid_indices):
            seg_windows[windows[win_idx].stt_idx].append(pos)

        norms = np.array([np.linalg.norm(emb) for emb in embeddings])
        keep = set(range(len(valid_indices)))
        utterance_count = 0
        total_discarded = 0

        for stt_idx, positions in seg_windows.items():
            if len(positions) <= 3:
                continue
            utterance_count += 1
            local_norms = norms[positions]
            median_norm = float(np.median(local_norms))
            mad = float(np.median(np.abs(local_norms - median_norm)))
            if mad < 1e-10:
                continue
            threshold = median_norm - self.config.norm_filter_sigma * 1.4826 * mad
            if threshold <= 0:
                continue
            for pos in positions:
                if norms[pos] < threshold:
                    keep.discard(pos)
                    total_discarded += 1

        if total_discarded > 0:
            print(
                f"  [Speakers] Per-utterance norm filter: discarded {total_discarded} embedding(s) "
                f"from {utterance_count} utterance(s) (k={self.config.norm_filter_sigma})"
            )

        kept_indices = [valid_indices[i] for i in sorted(keep)]
        kept_embeddings = [embeddings[i] for i in sorted(keep)]
        return kept_indices, kept_embeddings

    def _filter_by_utterance_coherence(
        self,
        valid_indices: list[int],
        embeddings: list[np.ndarray],
        windows: list[EmbeddingWindow],
        stt: STTTranscript,
    ) -> tuple[list[int], list[np.ndarray]]:
        if not embeddings or self.config.coherence_threshold <= 0.0:
            return valid_indices, embeddings
        threshold = self.config.coherence_threshold
        seg_positions: dict[int, list[int]] = defaultdict(list)
        for pos in range(len(valid_indices)):
            win = windows[valid_indices[pos]]
            seg_positions[win.stt_idx].append(pos)

        keep = set(range(len(valid_indices)))
        total_discarded = 0
        filtered_utterances = 0

        for stt_idx, positions in seg_positions.items():
            if len(positions) <= 1:
                continue
            embs = [embeddings[p] for p in positions]
            n = len(embs)
            sim_matrix = np.zeros((n, n), dtype=np.float64)
            for i in range(n):
                sim_matrix[i, i] = 1.0
                for j in range(i + 1, n):
                    ni = np.linalg.norm(embs[i])
                    nj = np.linalg.norm(embs[j])
                    if ni > 0 and nj > 0:
                        sim = float(np.dot(embs[i], embs[j]) / (ni * nj))
                    else:
                        sim = 0.0
                    sim_matrix[i, j] = sim
                    sim_matrix[j, i] = sim

            avg_sims = sim_matrix.sum(axis=1) / n
            order = sorted(range(n), key=lambda k: -avg_sims[k])

            coherent: list[int] = []
            for k in order:
                if all(sim_matrix[k, c] >= threshold for c in coherent):
                    coherent.append(k)

            if len(coherent) < n:
                filtered_utterances += 1
                coherent_set = set(coherent)
                for i in range(n):
                    if i not in coherent_set:
                        keep.discard(positions[i])
                        total_discarded += 1

        if total_discarded > 0:
            print(
                f"  [Speakers] Per-utterance coherence filter: discarded {total_discarded} embedding(s) "
                f"from {filtered_utterances} utterance(s) (threshold={threshold})"
            )

        kept_indices = [valid_indices[i] for i in sorted(keep)]
        kept_embeddings = [embeddings[i] for i in sorted(keep)]
        return kept_indices, kept_embeddings

    def _filter_by_utterance_low_coherence(
        self,
        valid_indices: list[int],
        embeddings: list[np.ndarray],
        windows: list[EmbeddingWindow],
        stt: STTTranscript,
    ) -> tuple[list[int], list[np.ndarray]]:
        if not embeddings or self.config.min_utterance_coherence_cluster <= 0.0:
            return valid_indices, embeddings
        threshold = self.config.min_utterance_coherence_cluster
        seg_positions: dict[int, list[int]] = defaultdict(list)
        for pos in range(len(valid_indices)):
            win = windows[valid_indices[pos]]
            seg_positions[win.stt_idx].append(pos)

        keep = set(range(len(valid_indices)))
        total_discarded = 0
        filtered_utterances = 0

        for stt_idx, positions in seg_positions.items():
            if len(positions) < 2:
                coherence = 1.0
            else:
                coherence = self._utterance_coherence(
                    [embeddings[p] for p in positions]
                )
            if coherence < threshold:
                for pos in positions:
                    keep.discard(pos)
                total_discarded += len(positions)
                filtered_utterances += 1

        if filtered_utterances > 0:
            print(
                f"  [Speakers] Low-coherence filter: discarded {total_discarded} window(s) from "
                f"{filtered_utterances} utterance(s) (threshold={threshold})"
            )

        kept_indices = [valid_indices[i] for i in sorted(keep)]
        kept_embeddings = [embeddings[i] for i in sorted(keep)]
        return kept_indices, kept_embeddings

    def _cluster_windows(
        self, X: np.ndarray, windows: list[EmbeddingWindow], valid_indices: list[int]
    ) -> np.ndarray:
        method = self.config.clustering_method
        seg_ids, utterance_embs = self._build_utterance_embeddings(
            X, windows, valid_indices
        )
        seg_to_pos = {seg_id: i for i, seg_id in enumerate(seg_ids)}

        if len(utterance_embs) == 1:
            utt_labels = np.array([1], dtype=np.int64)
        elif method == "umap_hdbscan_manual":
            utt_labels, mean_persistence = self._run_umap_hdbscan(
                utterance_embs,
                n_neighbors=self.config.umap_n_neighbors,
                min_cluster_size=self.config.hdbscan_min_cluster_size,
                n_components=self.config.umap_n_components,
            )
            n_clusters = len(set(int(x) for x in utt_labels if int(x) >= 0))
            n_noise = int(np.sum(utt_labels < 0))
            print(
                f"  [Speakers] Manual UMAP+HDBSCAN: {n_clusters} cluster(s), {n_noise} noise "
                f"(nn={self.config.umap_n_neighbors}, md={self.config.umap_min_dist}, "
                f"dim={self.config.umap_n_components}, mcs={self.config.hdbscan_min_cluster_size}, "
                f"persist={mean_persistence:.2f})"
            )
        elif method == "umap_hdbscan_auto":
            utt_labels = self._cluster_utterance_umap_hdbscan_auto(
                utterance_embs=utterance_embs,
                window_X=X,
                windows=windows,
                valid_indices=valid_indices,
                seg_ids=seg_ids,
            )
        else:
            raise ValueError(
                "Unsupported clustering method: "
                f"{method}. Supported: umap_hdbscan_auto, umap_hdbscan_manual"
            )

        win_labels = np.zeros((len(valid_indices),), dtype=utt_labels.dtype)
        for pos, win_idx in enumerate(valid_indices):
            seg_id = windows[win_idx].stt_idx
            win_labels[pos] = utt_labels[seg_to_pos[seg_id]]
        return win_labels

    def _build_utterance_profiles(
        self,
        X: np.ndarray,
        windows: list[EmbeddingWindow],
        valid_indices: list[int],
    ) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray]:
        seg_emb_accum: dict[int, list[np.ndarray]] = defaultdict(list)
        for pos, win_idx in enumerate(valid_indices):
            win = windows[win_idx]
            seg_emb_accum[win.stt_idx].append(X[pos])

        seg_ids = sorted(seg_emb_accum.keys())
        utterance_embs_raw = np.stack(
            [
                self._aggregate_utterance_embeddings(seg_emb_accum[seg_id])
                for seg_id in seg_ids
            ]
        )
        norms = np.linalg.norm(utterance_embs_raw, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        utterance_embs = utterance_embs_raw / norms
        window_counts = np.array(
            [len(seg_emb_accum[seg_id]) for seg_id in seg_ids], dtype=np.int64
        )
        coherence_scores = np.array(
            [
                SpeakerAttributor._utterance_coherence(seg_emb_accum[seg_id])
                for seg_id in seg_ids
            ],
            dtype=np.float64,
        )
        return seg_ids, utterance_embs, window_counts, coherence_scores

    def _build_utterance_embeddings(
        self,
        X: np.ndarray,
        windows: list[EmbeddingWindow],
        valid_indices: list[int],
    ) -> tuple[list[int], np.ndarray]:
        seg_ids, utterance_embs, _, _ = self._build_utterance_profiles(
            X, windows, valid_indices
        )
        return seg_ids, utterance_embs

    def _aggregate_utterance_embeddings(
        self, embeddings: list[np.ndarray]
    ) -> np.ndarray:
        if self.config.utterance_aggregation == "mean":
            return self._utterance_mean(embeddings)
        return self._utterance_medoid(embeddings)

    @staticmethod
    def _utterance_mean(embeddings: list[np.ndarray]) -> np.ndarray:
        if len(embeddings) == 1:
            return embeddings[0]
        return np.mean(np.stack(embeddings), axis=0)

    @staticmethod
    def _utterance_medoid(embeddings: list[np.ndarray]) -> np.ndarray:
        if len(embeddings) == 1:
            return embeddings[0]

        X = np.stack(embeddings)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        Xn = X / norms
        sim_matrix = Xn @ Xn.T
        avg_sims = sim_matrix.mean(axis=1)
        best_idx = int(np.argmax(avg_sims))
        return X[best_idx]

    @staticmethod
    def _utterance_coherence(embeddings: list[np.ndarray]) -> float:
        if len(embeddings) <= 1:
            return 1.0
        X = np.stack(embeddings)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        Xn = X / norms
        sim_matrix = Xn @ Xn.T
        avg_sims = (sim_matrix.sum(axis=1) - 1.0) / max(len(embeddings) - 1, 1)
        return float(np.max(avg_sims))

    def _run_umap_hdbscan(
        self,
        X: np.ndarray,
        *,
        n_neighbors: int,
        min_cluster_size: int,
        n_components: int,
    ) -> tuple[np.ndarray, float]:
        import hdbscan
        import umap

        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=self.config.umap_min_dist,
            n_components=n_components,
            metric="euclidean",
            random_state=42,
            n_jobs=1,
        )
        X_umap = reducer.fit_transform(X)
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=max(min_cluster_size - 1, 1),
            cluster_selection_method="eom",
        )
        labels = clusterer.fit_predict(X_umap)
        persistence = getattr(clusterer, "cluster_persistence_", None)
        mean_persistence = (
            float(np.mean(persistence))
            if persistence is not None and len(persistence)
            else 0.0
        )
        return labels, mean_persistence

    @staticmethod
    def _umap_sweep_candidates(n_items: int) -> list[tuple[int, int, int]]:
        if n_items <= 3:
            return []
        neighbor_values = {5, 10, 15, 25}
        mcs_values = {5, 8, 10}
        dim_values = {5, 10, 15}
        return sorted(
            (nn, mcs, dim)
            for nn in neighbor_values
            for mcs in mcs_values
            for dim in dim_values
        )

    @staticmethod
    def _compute_sim_matrix(X: np.ndarray) -> np.ndarray:
        """Row-normalized dot-product similarity (cosine sim)."""
        Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-10)
        return Xn @ Xn.T


    @staticmethod
    def _label_nn_purity(
        X: np.ndarray,
        labels: np.ndarray,
        *,
        sim_matrix: np.ndarray | None = None,
    ) -> float:
        if len(X) <= 1:
            return 1.0
        if sim_matrix is None:
            sim_matrix = SpeakerAttributor._compute_sim_matrix(X)
        n = len(sim_matrix)
        # Find nearest neighbor excluding self without mutating sim_matrix:
        # take the top-2 indices per row via argpartition and pick the
        # non-self one.
        top2 = np.argpartition(-sim_matrix, 1, axis=1)[:, :2]
        rows = np.arange(n)
        is_self = top2[:, 0] == rows
        nn_idx = np.where(is_self, top2[:, 1], top2[:, 0])
        valid = labels >= 0
        if not np.any(valid):
            return 0.0
        same = labels[valid] == labels[nn_idx[valid]]
        return float(np.mean(same)) if len(same) else 0.0

    @staticmethod
    def _singleton_point_fraction(labels: np.ndarray) -> float:
        """Fraction of non-noise *points* that live in singleton clusters.

        Point-weighted so it is comparable with the other score terms (all of
        which are computed per point, not per cluster).
        """
        valid_labels = labels[labels >= 0]
        if len(valid_labels) == 0:
            return 0.0
        _, counts = np.unique(valid_labels, return_counts=True)
        singleton_points = int(np.sum(counts[counts == 1]))
        return float(singleton_points / len(valid_labels))

    def _evaluate_clustering(
        self,
        utterance_X: np.ndarray,
        utt_labels: np.ndarray,
        window_X: np.ndarray,
        windows: list[EmbeddingWindow],
        valid_indices: list[int],
        seg_ids: list[int],
        mean_persistence: float,
        *,
        utterance_sim_matrix: np.ndarray | None = None,
        window_sim_matrix: np.ndarray | None = None,
    ) -> ClusteringScoreBreakdown:
        if utterance_sim_matrix is None:
            utterance_sim_matrix = self._compute_sim_matrix(utterance_X)
        if window_sim_matrix is None:
            window_sim_matrix = self._compute_sim_matrix(window_X)

        disagreement_rate, voted_separation = self._vote_consistency_metrics(
            window_X,
            windows,
            valid_indices,
            seg_ids,
            utt_labels,
            sim_matrix=window_sim_matrix,
        )
        nn_purity = self._label_nn_purity(
            utterance_X, utt_labels, sim_matrix=utterance_sim_matrix
        )
        singleton_point_fraction = self._singleton_point_fraction(utt_labels)
        silhouette = self._mean_silhouette(
            utterance_X, utt_labels, sim_matrix=utterance_sim_matrix
        )

        n_total = len(utt_labels)
        valid_mask = utt_labels >= 0
        n_valid = int(np.sum(valid_mask))
        noise_fraction = float(n_total - n_valid) / n_total if n_total > 0 else 0.0

        n_clusters = len(set(int(x) for x in utt_labels if x >= 0))
        max_cluster_ratio = 0.0
        if n_clusters > 1 and n_valid > 0:
            _, cluster_counts = np.unique(utt_labels[valid_mask], return_counts=True)
            max_cluster_ratio = float(cluster_counts.max()) / n_valid

        return ClusteringScoreBreakdown(
            score=0.0,
            silhouette=silhouette,
            nn_purity=nn_purity,
            disagreement_rate=disagreement_rate,
            voted_separation=voted_separation,
            singleton_point_fraction=singleton_point_fraction,
            noise_fraction=noise_fraction,
            n_clusters=n_clusters,
            max_cluster_ratio=max_cluster_ratio,
            mean_persistence=float(min(max(mean_persistence, 0.0), 1.0)),
        )

    def _vote_consistency_metrics(
        self,
        window_X: np.ndarray,
        windows: list[EmbeddingWindow],
        valid_indices: list[int],
        seg_ids: list[int],
        utt_labels: np.ndarray,
        *,
        sim_matrix: np.ndarray | None = None,
    ) -> tuple[float, float]:
        if len(window_X) == 0 or len(utt_labels) == 0:
            return 0.0, 0.0

        seg_to_pos = {seg_id: pos for pos, seg_id in enumerate(seg_ids)}
        expanded_labels = np.array(
            [
                int(utt_labels[seg_to_pos[windows[win_idx].stt_idx]])
                for win_idx in valid_indices
            ],
            dtype=np.int64,
        )
        valid_mask = expanded_labels >= 0
        if not np.any(valid_mask):
            return 0.0, 0.0

        cluster_ids = np.array(
            sorted(set(int(x) for x in expanded_labels if int(x) >= 0)), dtype=np.int64
        )
        centroids_raw = np.stack(
            [np.mean(window_X[expanded_labels == cid], axis=0) for cid in cluster_ids]
        )
        centroids = centroids_raw / np.maximum(
            np.linalg.norm(centroids_raw, axis=1, keepdims=True), 1e-10
        )
        window_sims = window_X @ centroids.T
        best_idx = np.argmax(window_sims, axis=1)
        best_labels = cluster_ids[best_idx]
        best_scores = window_sims[np.arange(len(window_X)), best_idx]

        seg_vote_weights: dict[int, dict[int, float]] = defaultdict(dict)
        seg_positions: dict[int, list[int]] = defaultdict(list)
        for pos, win_idx in enumerate(valid_indices):
            seg_id = windows[win_idx].stt_idx
            seg_positions[seg_id].append(pos)
            label = int(best_labels[pos])
            current = seg_vote_weights[seg_id].get(label, 0.0)
            seg_vote_weights[seg_id][label] = current + float(best_scores[pos])

        voted_by_seg: dict[int, int] = {}
        for seg_id, label_weights in seg_vote_weights.items():
            voted_by_seg[seg_id] = max(
                label_weights, key=lambda label: label_weights[label]
            )

        disagreement_count = 0
        voted_window_labels = np.full((len(window_X),), -1, dtype=np.int64)
        for seg_id, poses in seg_positions.items():
            voted_label = voted_by_seg.get(seg_id)
            if voted_label is None:
                continue
            voted_window_labels[poses] = voted_label
            disagreement_count += sum(
                1 for pos in poses if int(best_labels[pos]) != voted_label
            )

        disagreement_rate = (
            float(disagreement_count / len(window_X)) if len(window_X) > 0 else 0.0
        )
        voted_separation = self._mean_silhouette(
            window_X, voted_window_labels, sim_matrix=sim_matrix
        )
        return disagreement_rate, voted_separation

    def _build_sweep_candidate(
        self,
        *,
        labels: np.ndarray,
        utterance_X: np.ndarray,
        window_X: np.ndarray,
        windows: list[EmbeddingWindow],
        valid_indices: list[int],
        seg_ids: list[int],
        mean_persistence: float,
        top_str: str,
        nn: int,
        dim: int,
        mcs: int,
        utterance_sim_matrix: np.ndarray,
        window_sim_matrix: np.ndarray,
    ) -> SweepCandidateRow:
        breakdown = self._evaluate_clustering(
            utterance_X,
            labels,
            window_X,
            windows,
            valid_indices,
            seg_ids,
            mean_persistence,
            utterance_sim_matrix=utterance_sim_matrix,
            window_sim_matrix=window_sim_matrix,
        )
        return {
            "nn": nn,
            "dim": dim,
            "mcs": mcs,
            "score": 0.0,
            "clusters": breakdown.n_clusters,
            "n_noise": int(np.sum(labels < 0)),
            "top_str": top_str,
            "labels": labels,
            "nn_purity": breakdown.nn_purity,
            "disagree": breakdown.disagreement_rate,
            "sep": breakdown.voted_separation,
            "silhouette": breakdown.silhouette,
            "singleton": breakdown.singleton_point_fraction,
            "noise": breakdown.noise_fraction,
            "coarse": breakdown.max_cluster_ratio,
            "persistence": breakdown.mean_persistence,
        }

    @staticmethod
    def _top_k_sizes(labels: np.ndarray, ks: tuple[int, ...] = (1, 5, 10, 20, 30)) -> str:
        valid = labels[labels >= 0] if isinstance(labels, np.ndarray) else labels
        counts = sorted(Counter(int(x) for x in valid).values(), reverse=True)
        parts: list[str] = []
        for k in ks:
            if k <= len(counts):
                parts.append(f"top{k}={counts[k - 1]:4d}")
            else:
                parts.append(f"top{k}=   -")
        return " ".join(parts)

    def _cluster_utterance_umap_hdbscan_auto(
        self,
        *,
        utterance_embs: np.ndarray,
        window_X: np.ndarray,
        windows: list[EmbeddingWindow],
        valid_indices: list[int],
        seg_ids: list[int],
    ) -> np.ndarray:
        candidates = self._umap_sweep_candidates(len(utterance_embs))
        if not candidates:
            labels, mean_persistence = self._run_umap_hdbscan(
                utterance_embs,
                n_neighbors=min(max(2, len(utterance_embs) - 1), 5),
                min_cluster_size=2,
                n_components=min(5, max(2, utterance_embs.shape[1])),
            )
            n_clusters = len(set(int(x) for x in labels if int(x) >= 0))
            n_noise = int(np.sum(labels < 0))
            print(
                "  [Speakers] Auto UMAP+HDBSCAN fallback: "
                f"{n_clusters} cluster(s), {n_noise} noise "
                f"(persist={mean_persistence:.2f})"
            )
            return labels

        utt_scoring_sim = self._compute_sim_matrix(utterance_embs)
        win_scoring_sim = self._compute_sim_matrix(window_X)
        rows: list[SweepCandidateRow] = []
        for nn, mcs, dim in candidates:
            labels, mean_persistence = self._run_umap_hdbscan(
                utterance_embs,
                n_neighbors=nn,
                min_cluster_size=mcs,
                n_components=dim,
            )
            rows.append(
                self._build_sweep_candidate(
                    labels=labels,
                    utterance_X=utterance_embs,
                    window_X=window_X,
                    windows=windows,
                    valid_indices=valid_indices,
                    seg_ids=seg_ids,
                    mean_persistence=mean_persistence,
                    top_str=self._top_k_sizes(labels),
                    nn=nn,
                    dim=dim,
                    mcs=mcs,
                    utterance_sim_matrix=utt_scoring_sim,
                    window_sim_matrix=win_scoring_sim,
                )
            )

        self._assign_normalized_candidate_scores(rows)
        aggregation = self.config.utterance_aggregation
        print(f"  [Speakers] Auto UMAP+HDBSCAN (utterance-{aggregation}):")
        for row in rows:
            print(
                f"    nn={row['nn']:2d} dim={row['dim']} mcs={row['mcs']:2d} "
                f"→ {row['clusters']:3d} cluster(s), {row['n_noise']:4d} noise, "
                f"score={row['score']:.2f}, disagree={row['disagree']:.3f}, sep={row['sep']:.3f} "
                f"{row['top_str']}"
            )
        self._print_score_usefulness(rows)

        selected = max(
            rows,
            key=lambda row: (
                float(row["score"]),
                float(row["sep"]),
                -float(row["disagree"]),
                float(row["nn_purity"]),
                float(row["silhouette"]),
                -float(row["n_noise"]),
            ),
        )
        labels = np.asarray(selected["labels"], dtype=np.int64)
        n_clusters = len(set(int(x) for x in labels if int(x) >= 0))
        n_noise = int(np.sum(labels < 0))
        print(
            "  [Speakers] Auto UMAP+HDBSCAN: "
            f"{n_clusters} cluster(s), {n_noise} noise "
            f"(nn={selected['nn']}, dim={selected['dim']}, mcs={selected['mcs']}, "
            f"persist={selected['persistence']:.2f}, score={selected['score']:.2f}, "
            f"disagree={selected['disagree']:.3f}, sep={selected['sep']:.3f})"
        )
        return labels

    @staticmethod
    def _mean_silhouette(
        X: np.ndarray,
        labels: np.ndarray,
        *,
        sim_matrix: np.ndarray | None = None,
    ) -> float:
        """Vectorized mean silhouette in cosine space over non-noise points.

        Uses sklearn's C-implemented silhouette_score with a precomputed
        cosine distance matrix. Noise-labeled points (-1) are excluded before
        scoring so they don't become a spurious "noise cluster."
        """
        valid_mask = labels >= 0
        if valid_mask.sum() < 2:
            return 0.0
        valid_labels = labels[valid_mask]
        unique_valid = set(int(x) for x in valid_labels)
        if len(unique_valid) < 2:
            return 0.0
        if sim_matrix is None:
            sim_matrix = SpeakerAttributor._compute_sim_matrix(X)

        from sklearn.metrics import silhouette_score

        idx = np.where(valid_mask)[0]
        dist_valid = 1.0 - sim_matrix[np.ix_(idx, idx)]
        np.clip(dist_valid, 0.0, None, out=dist_valid)
        np.fill_diagonal(dist_valid, 0.0)
        try:
            return float(
                silhouette_score(dist_valid, valid_labels, metric="precomputed")
            )
        except ValueError:
            return 0.0

    @staticmethod
    def _metric_rank(
        candidates: list[SweepCandidateRow],
        selected: SweepCandidateRow,
        key: str,
        *,
        reverse: bool,
    ) -> int:
        ordered = sorted(
            candidates,
            key=lambda row: float(cast(dict[str, Any], row)[key]),
            reverse=reverse,
        )
        for rank, row in enumerate(ordered, start=1):
            if row is selected:
                return rank
        return len(candidates)

    @staticmethod
    def _normalized_metric(
        values: list[float], *, higher_is_better: bool
    ) -> list[float]:
        if not values:
            return []
        arr = np.array(values, dtype=np.float64)
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        if std < 1e-12:
            return [0.0] * len(values)
        norm = (arr - mean) / std
        if not higher_is_better:
            norm = -norm
        return [float(x) for x in norm]

    def _assign_normalized_candidate_scores(
        self, candidates: list[SweepCandidateRow]
    ) -> None:
        if not candidates:
            return

        specs = [
            ("silhouette", 1.0, True),
            ("nn_purity", 1.0, True),
            ("sep", 1.0, True),
            ("disagree", 1.0, False),
        ]

        normalized_by_key: dict[str, list[float]] = {}
        for key, _weight, higher_is_better in specs:
            normalized_by_key[key] = self._normalized_metric(
                [float(cast(dict[str, Any], row)[key]) for row in candidates],
                higher_is_better=higher_is_better,
            )

        for idx, row in enumerate(candidates):
            row["score"] = float(
                sum(weight * normalized_by_key[key][idx] for key, weight, _hib in specs)
            )

    def _print_score_usefulness(self, candidates: list[SweepCandidateRow]) -> None:
        if not candidates:
            return

        best = max(candidates, key=lambda row: float(row["score"]))
        best_nn = max(candidates, key=lambda row: float(row["nn_purity"]))
        best_sep = max(candidates, key=lambda row: float(row["sep"]))
        best_sil = max(candidates, key=lambda row: float(row["silhouette"]))
        best_disagree = min(candidates, key=lambda row: float(row["disagree"]))

        def _fmt(row: SweepCandidateRow) -> str:
            return f"nn={row['nn']:2d} dim={row['dim']} mcs={row['mcs']:2d}"

        print("    usefulness:")
        print(f"      selected: {_fmt(best)}")
        print(
            f"      best nn_purity: {_fmt(best_nn)} ({float(best_nn['nn_purity']):.3f})"
        )
        print(
            f"      best disagree:  {_fmt(best_disagree)} ({float(best_disagree['disagree']):.3f})"
        )
        print(f"      best sep:       {_fmt(best_sep)} ({float(best_sep['sep']):.3f})")
        print(
            f"      best silhouette:{_fmt(best_sil)} ({float(best_sil['silhouette']):.3f})"
        )
        print(
            "      selected ranks:"
            f" nn={self._metric_rank(candidates, best, 'nn_purity', reverse=True)}/{len(candidates)}"
            f" disagree={self._metric_rank(candidates, best, 'disagree', reverse=False)}/{len(candidates)}"
            f" sep={self._metric_rank(candidates, best, 'sep', reverse=True)}/{len(candidates)}"
            f" silhouette={self._metric_rank(candidates, best, 'silhouette', reverse=True)}/{len(candidates)}"
        )
        print(
            "      spans:"
            f" nn={min(row['nn_purity'] for row in candidates):.3f}..{max(row['nn_purity'] for row in candidates):.3f}"
            f" disagree={min(row['disagree'] for row in candidates):.3f}..{max(row['disagree'] for row in candidates):.3f}"
            f" sep={min(row['sep'] for row in candidates):.3f}..{max(row['sep'] for row in candidates):.3f}"
            f" silhouette={min(row['silhouette'] for row in candidates):.3f}..{max(row['silhouette'] for row in candidates):.3f}"
            f" coarse={min(row['coarse'] for row in candidates):.3f}..{max(row['coarse'] for row in candidates):.3f}"
            f" noise={min(row['noise'] for row in candidates):.3f}..{max(row['noise'] for row in candidates):.3f}"
        )

    def _vote_speakers(
        self,
        stt: STTTranscript,
        windows: list[EmbeddingWindow],
        valid_indices_all: list[int],
        embeddings_all: list[np.ndarray],
        speaker_centroids: dict[str, np.ndarray],
    ) -> list[SpeakerAttributionEntry]:
        """Assign speakers to utterances via per-window centroid similarity voting.

        Important: votes are computed from *all embedded windows* (pre-filter).
        Filtering only affects how centroids are formed.
        """

        segment_votes: dict[int, list[tuple[str, float]]] = defaultdict(list)
        segment_window_votes: dict[int, list[WindowVote]] = defaultdict(list)
        embedded_windows_by_seg: dict[int, int] = defaultdict(int)
        best_sim_by_seg: dict[int, float] = {}

        centroid_ids = list(speaker_centroids.keys())
        centroid_mat = (
            np.stack([speaker_centroids[sid] for sid in centroid_ids])
            if centroid_ids
            else None
        )
        min_sim = float(self.config.min_vote_similarity)

        for pos, win_idx in enumerate(valid_indices_all):
            win = windows[win_idx]
            seg_idx = win.stt_idx
            embedded_windows_by_seg[seg_idx] += 1

            if centroid_mat is None:
                continue

            emb = embeddings_all[pos]
            sims = centroid_mat @ emb
            best_i = int(np.argmax(sims))
            best_sim = float(sims[best_i])

            prev_best = best_sim_by_seg.get(seg_idx)
            if prev_best is None or best_sim > prev_best:
                best_sim_by_seg[seg_idx] = best_sim

            if best_sim < min_sim:
                continue

            speaker_id = centroid_ids[best_i]
            segment_votes[seg_idx].append((speaker_id, best_sim))
            segment_window_votes[seg_idx].append(
                WindowVote(
                    speaker_id=speaker_id,
                    similarity=best_sim,
                    start=win.start,
                    end=win.end,
                )
            )

        voted_segments: dict[int, str] = {}
        voted_details: dict[int, tuple[str, float | None, bool]] = {}

        for idx in range(len(stt.entries)):
            votes = segment_votes.get(idx)
            if votes:
                speaker_weights: dict[str, float] = defaultdict(float)
                for speaker_id, weight in votes:
                    speaker_weights[speaker_id] += weight
                winner = max(speaker_weights, key=lambda k: speaker_weights[k])
                total_weight = sum(w for s, w in votes if s == winner)
                num_windows = sum(1 for s, _ in votes if s == winner)
                avg_similarity = total_weight / num_windows if num_windows > 0 else None

                voted_segments[idx] = winner
                voted_details[idx] = (winner, avg_similarity, len(votes) >= 2)

        entries: list[SpeakerAttributionEntry] = []
        for idx in range(len(stt.entries)):
            entry = stt.entries[idx]
            entry_id = self._stt_entry_id(entry, idx)
            embedded_count = int(embedded_windows_by_seg.get(idx, 0))
            best_sim = best_sim_by_seg.get(idx)

            if idx in voted_details:
                winner, avg_similarity, enrolled = voted_details[idx]
                entries.append(
                    SpeakerAttributionEntry(
                        entry_id=entry_id,
                        start=entry.start,
                        end=entry.end,
                        speaker_id=winner,
                        text=entry.text,
                        similarity=avg_similarity,
                        embedding_present=embedded_count > 0,
                        enrolled=enrolled,
                        window_votes=segment_window_votes.get(idx, []),
                    )
                )
                continue

            # Fallback by time only for utterances with <=1 embedded window AND very low similarity.
            if embedded_count <= 1 and (best_sim is None or best_sim < min_sim):
                speaker_id = self._nearest_speaker_by_time_from_voted(
                    idx, voted_segments, stt
                )
            else:
                speaker_id = "UNKNOWN"

            entries.append(
                SpeakerAttributionEntry(
                    entry_id=entry_id,
                    start=entry.start,
                    end=entry.end,
                    speaker_id=speaker_id,
                    text=entry.text,
                    similarity=best_sim,
                    embedding_present=embedded_count > 0,
                    enrolled=False,
                    window_votes=segment_window_votes.get(idx, []),
                )
            )

        return entries

    @staticmethod
    def _stt_entry_id(entry: Any, idx: int) -> str:
        raw = getattr(entry, "entry_id", "") or ""
        value = str(raw).strip()
        if value:
            return value
        # Backward compatibility for legacy stt.json files without entry_id.
        return f"utt_{idx:06d}"

    def _compute_cluster_quality(
        self,
        attribution_entries: list[SpeakerAttributionEntry],
        speaker_centroids: dict[str, np.ndarray],
        cluster_labels: np.ndarray,
        label_to_speaker: dict[int, str],
        X_cluster: np.ndarray,
        valid_indices_cluster: list[int],
        windows: list[EmbeddingWindow],
    ) -> list[SpeakerClusterQuality]:
        centroid_ids = sorted(speaker_centroids.keys())
        if len(centroid_ids) < 2:
            nearest_info: dict[str, tuple[str, float]] = {}
        else:
            centroid_mat = np.stack([speaker_centroids[sid] for sid in centroid_ids])
            centroid_sims = centroid_mat @ centroid_mat.T
            nearest_info = {}
            for i, sid in enumerate(centroid_ids):
                sims = centroid_sims[i].copy()
                sims[i] = -1.0
                best_j = int(np.argmax(sims))
                nearest_info[sid] = (centroid_ids[best_j], float(sims[best_j]))

        speaker_windows: dict[str, list[int]] = defaultdict(list)
        for pos in range(len(valid_indices_cluster)):
            label = int(cluster_labels[pos])
            if label < 0:
                continue
            speaker_id = label_to_speaker[label]
            speaker_windows[speaker_id].append(pos)

        utterance_by_speaker: dict[str, list[SpeakerAttributionEntry]] = defaultdict(
            list
        )
        for entry in attribution_entries:
            if entry.speaker_id != "UNKNOWN":
                utterance_by_speaker[entry.speaker_id].append(entry)

        results: list[SpeakerClusterQuality] = []
        for speaker_id in centroid_ids:
            entries = utterance_by_speaker.get(speaker_id, [])
            enrolled_entries = [e for e in entries if e.enrolled]
            num_windows = len(speaker_windows.get(speaker_id, []))

            enrolled_sims = [
                e.similarity for e in enrolled_entries if e.similarity is not None
            ]
            all_sims = [e.similarity for e in entries if e.similarity is not None]

            mean_sim: float | None = None
            median_sim: float | None = None
            p25_sim: float | None = None
            p75_sim: float | None = None
            min_sim: float | None = None
            max_sim: float | None = None

            if enrolled_sims:
                arr = np.array(enrolled_sims)
                mean_sim = float(arr.mean())
                median_sim = float(np.median(arr))
                p25_sim = float(np.percentile(arr, 25))
                p75_sim = float(np.percentile(arr, 75))
                min_sim = float(arr.min())
            if all_sims:
                max_sim = float(np.array(all_sims).max())

            intra_sim: float | None = None
            win_positions = speaker_windows.get(speaker_id, [])
            if len(win_positions) >= 2:
                # X_cluster is already L2-normalized upstream.
                win_embs = X_cluster[win_positions]
                sim_tri = win_embs @ win_embs.T
                idx_upper = np.triu_indices(len(win_positions), k=1)
                intra_sim = float(sim_tri[idx_upper].mean())

            nearest_sid, nearest_sim = nearest_info.get(speaker_id, (None, None))

            results.append(
                SpeakerClusterQuality(
                    speaker_id=speaker_id,
                    num_utterances=len(entries),
                    num_windows=num_windows,
                    num_enrolled=len(enrolled_entries),
                    mean_similarity=mean_sim,
                    median_similarity=median_sim,
                    p25_similarity=p25_sim,
                    p75_similarity=p75_sim,
                    min_similarity=min_sim,
                    max_similarity=max_sim,
                    intra_cluster_similarity=intra_sim,
                    nearest_speaker=nearest_sid,
                    nearest_speaker_similarity=nearest_sim,
                )
            )

        return results

    def _print_diagnostics(
        self,
        X: np.ndarray,
        windows: list[EmbeddingWindow],
        valid_indices: list[int],
    ) -> None:
        # Note: intentionally noisy/expensive diagnostics are opt-in.
        if len(X) <= 1:
            return

        distances = pdist(X, metric="cosine")
        all_sims = 1.0 - distances
        print(f"  [Speakers] Pairwise similarity ({len(all_sims)} pairs):")
        print(
            f"    min={all_sims.min():.4f} max={all_sims.max():.4f} "
            f"mean={all_sims.mean():.4f} median={np.median(all_sims):.4f}"
        )
        for p in [5, 10, 25, 50, 75, 90, 95]:
            print(f"    p{p}={np.percentile(all_sims, p):.4f}")

        within_sims: list[float] = []
        between_sims: list[float] = []
        seg_windows: dict[int, list[int]] = defaultdict(list)
        for pos, win_idx in enumerate(valid_indices):
            seg_windows[windows[win_idx].stt_idx].append(pos)
        same_set: set[tuple[int, int]] = set()
        for poses in seg_windows.values():
            for i in range(len(poses)):
                for j in range(i + 1, len(poses)):
                    same_set.add(
                        (poses[i], poses[j])
                        if poses[i] < poses[j]
                        else (poses[j], poses[i])
                    )

        from scipy.spatial.distance import squareform

        sim_matrix = 1.0 - squareform(distances) if len(X) > 1 else np.array([[]])
        if sim_matrix.size > 0:
            n = len(X)
            for i in range(n):
                for j in range(i + 1, n):
                    s = float(sim_matrix[i, j])
                    if (i, j) in same_set:
                        within_sims.append(s)
                    else:
                        between_sims.append(s)

        if within_sims:
            wa = np.array(within_sims)
            print(
                f"    WITHIN-segment ({len(wa)} pairs): "
                f"min={wa.min():.4f} max={wa.max():.4f} "
                f"mean={wa.mean():.4f} median={np.median(wa):.4f}"
            )
            for p in [5, 25, 50, 75, 95]:
                print(f"      p{p}={np.percentile(wa, p):.4f}")
        if between_sims:
            ba = np.array(between_sims)
            print(
                f"    BETWEEN-segment ({len(ba)} pairs): "
                f"min={ba.min():.4f} max={ba.max():.4f} "
                f"mean={ba.mean():.4f} median={np.median(ba):.4f}"
            )

        if sim_matrix.size > 0 and len(seg_windows) > 1:
            n = len(X)
            same_count = 0
            diff_count = 0
            for i in range(n):
                sims_row = sim_matrix[i].copy()
                sims_row[i] = -1.0
                nn_idx = int(np.argmax(sims_row))
                i_seg = None
                nn_seg = None
                for seg_idx, poses in seg_windows.items():
                    if i in poses:
                        i_seg = seg_idx
                    if nn_idx in poses:
                        nn_seg = seg_idx
                if i_seg is not None and nn_seg is not None:
                    if i_seg == nn_seg:
                        same_count += 1
                    else:
                        diff_count += 1
            total = same_count + diff_count
            purity = same_count / total if total > 0 else 0.0
            print(
                f"    NN-purity: {purity:.4f} ({same_count}/{total} windows have nearest neighbor in same segment)"
            )
            if purity >= 0.8:
                print(
                    "           → GOOD: embeddings are discriminative for speaker clustering"
                )
            elif purity >= 0.5:
                print("           → MARGINAL: clustering may produce noisy results")
            else:
                print(
                    "           → POOR: embeddings lack speaker signal, clustering will not work reliably"
                )

    @staticmethod
    def _nearest_speaker_by_time_from_voted(
        target_idx: int,
        voted_segments: dict[int, str],
        stt: STTTranscript,
    ) -> str:
        if not voted_segments:
            return "UNKNOWN"
        target_time = stt.entries[target_idx].start
        best_idx: int | None = None
        best_dist = float("inf")
        for seg_idx in voted_segments:
            dist = abs(stt.entries[seg_idx].start - target_time)
            if dist < best_dist:
                best_dist = dist
                best_idx = seg_idx
        if best_idx is not None:
            return voted_segments[best_idx]
        return "UNKNOWN"

    def _unknown_attribution(
        self,
        stt: STTTranscript,
        output_dir: Path,
        bank: SpeakerBankManager,
    ) -> SpeakerAttribution:
        attribution_entries = [
            SpeakerAttributionEntry(
                entry_id=self._stt_entry_id(entry, idx),
                start=entry.start,
                end=entry.end,
                speaker_id="UNKNOWN",
                text=entry.text,
                similarity=None,
                embedding_present=False,
                enrolled=False,
                window_votes=[],
            )
            for idx, entry in enumerate(stt.entries)
        ]
        bank.save(output_dir)
        return SpeakerAttribution(entries=attribution_entries)
