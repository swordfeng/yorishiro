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
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

from yorishiro.audio.speaker_bank import SpeakerBankManager, SpeakerBankManagerConfig
from yorishiro.models.film_models import SpeakerAttribution, SpeakerAttributionEntry, STTTranscript, WindowVote


@dataclass(frozen=True)
class EmbeddingWindow:
    stt_idx: int
    start: float
    end: float


@dataclass(frozen=True)
class SpeakerAttributorConfig:
    embedding_backend: str = "wespeaker"
    similarity_threshold: float = 0.75
    diagnostics_enabled: bool = False
    clustering_method: str = "umap_hdbscan_utterance"
    window_duration: float = 1.5
    window_hop: float = 0.75
    min_window_duration: float = 0.8
    energy_threshold_db: float = -40.0
    norm_filter_sigma: float = 5.0
    coherence_threshold: float = 0.1
    # If a window's best centroid cosine similarity is below this threshold,
    # the window contributes no votes.
    min_vote_similarity: float = 0.2
    hdbscan_min_cluster_size: int = 10
    umap_n_neighbors: int = 5
    umap_min_dist: float = 0.0
    umap_n_components: int = 5
    hf_token_env: str = "YORISHIRO_HF_TOKEN"


class SpeakerAttributor:
    """Assign provisional speaker IDs to STT entries via windowed embedding + voting."""

    def __init__(self, config: SpeakerAttributorConfig | None = None) -> None:
        self.config = config or SpeakerAttributorConfig()

    def run(self, audio_path: Path, output_dir: Path) -> SpeakerAttribution:
        output_dir.mkdir(parents=True, exist_ok=True)
        stt = STTTranscript(**json.loads((output_dir / "stt.json").read_text(encoding="utf-8")))
        attribution = self._attribute(audio_path, stt, output_dir)
        out = output_dir / "speaker_attribution.json"
        out.write_text(attribution.model_dump_json(indent=2), encoding="utf-8")
        num_speakers = len({e.speaker_id for e in attribution.entries if e.speaker_id != "UNKNOWN"})
        print(f"  [Speakers] Done — {len(attribution.entries)} entry attribution(s), {num_speakers} speaker(s)")
        return attribution

    def _attribute(self, audio_path: Path, stt: STTTranscript, output_dir: Path) -> SpeakerAttribution:
        bank = SpeakerBankManager(
            SpeakerBankManagerConfig(
                embedding_backend=self.config.embedding_backend,
                hf_token_env=self.config.hf_token_env,
            )
        )

        # ── Phase 1: Extract windows ────────────────────────────────
        windows = self._extract_windows(stt)
        print(f"  [Speakers] {len(windows)} window(s) from {len(stt.entries)} segment(s)")

        if not windows:
            print("  [Speakers] No windows — assigning all entries as UNKNOWN")
            return self._unknown_attribution(stt, output_dir, bank)

        # ── Phase 2: Embed windows ───────────────────────────────────
        valid_indices_all, embeddings_all_raw = self._embed_windows(windows, audio_path, bank)

        print(f"  [Speakers] {len(embeddings_all_raw)} valid embedding(s) from {len(windows)} window(s)")

        if not embeddings_all_raw:
            print("  [Speakers] No valid embeddings — assigning all entries as UNKNOWN")
            return self._unknown_attribution(stt, output_dir, bank)

        # Normalize *all* embeddings for centroid assignment/voting.
        X_all_raw = np.stack(embeddings_all_raw)
        norms_all = np.linalg.norm(X_all_raw, axis=1, keepdims=True)
        norms_all = np.maximum(norms_all, 1e-10)
        X_all = X_all_raw / norms_all
        embeddings_all = [X_all[i] for i in range(len(valid_indices_all))]

        # ── Phase 2b-d: Filters for clustering only ─────────────────
        # We still assign/vote using *all* embedded windows (pre-filter).
        # Filters only decide which windows are trusted to form clusters/centroids.

        valid_indices_cluster, embeddings_cluster_raw = (
            list(valid_indices_all),
            list(embeddings_all_raw),
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

        # In-utterance coherence filter (on normalized embeddings).
        valid_indices_cluster, embeddings_cluster = self._filter_by_utterance_coherence(
            valid_indices_cluster, embeddings_cluster, windows, stt
        )

        if len(valid_indices_cluster) == 0:
            print("  [Speakers] No clusterable embeddings after filtering — assigning by temporal proximity only")
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
            speaker_id: np.mean(np.stack(embs), axis=0) for speaker_id, embs in speaker_emb_accum.items()
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
            if sid != "UNKNOWN" and (sid not in speaker_first_time or entry.start < speaker_first_time[sid]):
                speaker_first_time[sid] = entry.start

        for speaker_id in sorted(speaker_centroids.keys()):
            bank.speaker_bank.add_speaker(speaker_id, speaker_first_time.get(speaker_id, 0.0))
            bank._embeddings[speaker_id] = speaker_centroids[speaker_id]

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
                windows.append(EmbeddingWindow(stt_idx=idx, start=entry.start, end=entry.end))
                continue
            if duration < window_dur:
                windows.append(EmbeddingWindow(stt_idx=idx, start=entry.start, end=entry.end))
                continue

            t = entry.start
            while t + window_dur <= entry.end + 1e-6:
                win_end = min(t + window_dur, entry.end)
                windows.append(EmbeddingWindow(stt_idx=idx, start=t, end=win_end))
                t += window_hop

            tail_start = entry.end - window_dur
            if tail_start > t - window_hop + 1e-6 and tail_start >= entry.start:
                windows.append(EmbeddingWindow(stt_idx=idx, start=tail_start, end=entry.end))

        return windows

    def _window_has_energy(self, audio_path: Path, win: EmbeddingWindow) -> bool:
        try:
            info = sf.info(str(audio_path))
            start_sample = int(win.start * info.samplerate)
            end_sample = int(win.end * info.samplerate)
            chunk, _ = sf.read(str(audio_path), start=start_sample, stop=end_sample, dtype="float32", always_2d=False)
            if chunk.ndim > 1:
                chunk = chunk.mean(axis=1)
            rms = np.sqrt(np.mean(chunk ** 2))
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
    ) -> tuple[list[int], list[np.ndarray]]:
        valid_indices: list[int] = []
        embeddings: list[np.ndarray] = []

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

                emb = bank.extract_speaker_embedding(audio_path, win.start, win.end)
                if emb is not None:
                    valid_indices.append(i)
                    embeddings.append(emb)
                progress.update(1)

        return valid_indices, embeddings

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
            print(f"  [Speakers] Per-utterance norm filter: discarded {total_discarded} embedding(s) "
                  f"from {utterance_count} utterance(s) (k={self.config.norm_filter_sigma})")

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
            print(f"  [Speakers] Per-utterance coherence filter: discarded {total_discarded} embedding(s) "
                  f"from {filtered_utterances} utterance(s) (threshold={threshold})")

        kept_indices = [valid_indices[i] for i in sorted(keep)]
        kept_embeddings = [embeddings[i] for i in sorted(keep)]
        return kept_indices, kept_embeddings

    def _cluster_windows(self, X: np.ndarray, windows: list[EmbeddingWindow], valid_indices: list[int]) -> np.ndarray:
        method = self.config.clustering_method
        if method in {"umap_hdbscan_utterance", "umap_hdbscan_utt"}:
            # Average window embeddings per STT segment, cluster utterance embeddings,
            # then expand back to per-window labels for downstream voting/centroids.
            seg_emb_accum: dict[int, list[np.ndarray]] = defaultdict(list)
            for pos, win_idx in enumerate(valid_indices):
                win = windows[win_idx]
                seg_emb_accum[win.stt_idx].append(X[pos])

            seg_ids = sorted(seg_emb_accum.keys())
            seg_to_pos = {seg_id: i for i, seg_id in enumerate(seg_ids)}

            utterance_embs_raw = np.stack([np.mean(seg_emb_accum[seg_id], axis=0) for seg_id in seg_ids])
            norms = np.linalg.norm(utterance_embs_raw, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            utterance_embs = utterance_embs_raw / norms

            if len(utterance_embs) == 1:
                utt_labels = np.array([1])
            else:
                utt_labels = self._cluster(utterance_embs, method="umap_hdbscan")

            win_labels = np.zeros((len(valid_indices),), dtype=utt_labels.dtype)
            for pos, win_idx in enumerate(valid_indices):
                seg_id = windows[win_idx].stt_idx
                win_labels[pos] = utt_labels[seg_to_pos[seg_id]]
            return win_labels

        return self._cluster(X, method=method)

    def _cluster(self, X: np.ndarray, method: str | None = None) -> np.ndarray:
        method = method or self.config.clustering_method
        if method == "umap_hdbscan":
            import hdbscan
            import umap
            reducer = umap.UMAP(
                n_neighbors=self.config.umap_n_neighbors,
                min_dist=self.config.umap_min_dist,
                n_components=self.config.umap_n_components,
                metric="cosine",
                random_state=42,
                n_jobs=1,
            )
            X_umap = reducer.fit_transform(X)
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.config.hdbscan_min_cluster_size,
                # Use HDBSCAN's default behavior (min_samples=min_cluster_size)
                # by explicitly mirroring it. This is more conservative than
                # min_samples=mcs//2 and matches the expected cluster counts.
                min_samples=self.config.hdbscan_min_cluster_size,
                cluster_selection_method="eom",
            )
            labels = clusterer.fit_predict(X_umap)
            n_clusters = len(set(labels) - {-1})
            n_noise = int(np.sum(labels == -1))
            if n_clusters == 0:
                print("  [Speakers] UMAP+HDBSCAN found 0 clusters, falling back to HAC")
                distances = pdist(X, metric="cosine")
                Z = linkage(distances, method="average")
                threshold = 1.0 - self.config.similarity_threshold
                labels = fcluster(Z, t=threshold, criterion="distance")
            else:
                print(f"  [Speakers] UMAP+HDBSCAN: {n_clusters} cluster(s), {n_noise} noise "
                      f"(nn={self.config.umap_n_neighbors}, md={self.config.umap_min_dist}, "
                      f"dim={self.config.umap_n_components}, mcs={self.config.hdbscan_min_cluster_size})")
            return labels
        if method == "hdbscan":
            import hdbscan
            dist_matrix = squareform(pdist(X, metric="cosine"))
            clusterer = hdbscan.HDBSCAN(
                metric="precomputed",
                min_cluster_size=self.config.hdbscan_min_cluster_size,
                min_samples=self.config.hdbscan_min_cluster_size,
                cluster_selection_method="eom",
            )
            labels = clusterer.fit_predict(dist_matrix)
            n_clusters = len(set(labels) - {-1})
            n_noise = int(np.sum(labels == -1))
            if n_clusters == 0:
                print(f"  [Speakers] HDBSCAN found 0 clusters ({n_noise} noise), falling back to HAC")
                distances = pdist(X, metric="cosine")
                Z = linkage(distances, method="average")
                threshold = 1.0 - self.config.similarity_threshold
                labels = fcluster(Z, t=threshold, criterion="distance")
            else:
                print(f"  [Speakers] HDBSCAN: {n_clusters} cluster(s), {n_noise} noise point(s) "
                      f"(min_cluster_size={self.config.hdbscan_min_cluster_size})")
            return labels
        if method == "spectral":
            from sklearn.cluster import SpectralClustering
            distances = pdist(X, metric="cosine")
            n_clusters_est = max(2, min(30, len(X) // self.config.hdbscan_min_cluster_size))
            sc = SpectralClustering(
                n_clusters=n_clusters_est,
                affinity="precomputed",
                random_state=42,
            )
            sim_matrix = 1.0 - squareform(distances)
            np.fill_diagonal(sim_matrix, 1.0)
            sim_matrix = np.maximum(sim_matrix, 0.0)
            labels = sc.fit_predict(sim_matrix)
            n_found = len(set(labels))
            print(f"  [Speakers] Spectral: {n_found} cluster(s) (requested {n_clusters_est})")
            return labels
        distances = pdist(X, metric="cosine")
        Z = linkage(distances, method=method)
        threshold = 1.0 - self.config.similarity_threshold
        labels = fcluster(Z, t=threshold, criterion="distance")
        return labels

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
        centroid_mat = np.stack([speaker_centroids[sid] for sid in centroid_ids]) if centroid_ids else None
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
            entry_id = f"utt_{idx:06d}"
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
                speaker_id = self._nearest_speaker_by_time_from_voted(idx, voted_segments, stt)
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
        print(f"    min={all_sims.min():.4f} max={all_sims.max():.4f} "
              f"mean={all_sims.mean():.4f} median={np.median(all_sims):.4f}")
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
                    same_set.add((poses[i], poses[j]) if poses[i] < poses[j] else (poses[j], poses[i]))

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
            print(f"    WITHIN-segment ({len(wa)} pairs): "
                  f"min={wa.min():.4f} max={wa.max():.4f} "
                  f"mean={wa.mean():.4f} median={np.median(wa):.4f}")
            for p in [5, 25, 50, 75, 95]:
                print(f"      p{p}={np.percentile(wa, p):.4f}")
        if between_sims:
            ba = np.array(between_sims)
            print(f"    BETWEEN-segment ({len(ba)} pairs): "
                  f"min={ba.min():.4f} max={ba.max():.4f} "
                  f"mean={ba.mean():.4f} median={np.median(ba):.4f}")

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
            print(f"    NN-purity: {purity:.4f} ({same_count}/{total} windows have nearest neighbor in same segment)")
            if purity >= 0.8:
                print("           → GOOD: embeddings are discriminative for speaker clustering")
            elif purity >= 0.5:
                print("           → MARGINAL: clustering may produce noisy results")
            else:
                print("           → POOR: embeddings lack speaker signal, clustering will not work reliably")

        # ── UMAP+HDBSCAN (utterance-averaged) diagnostics ─────────────
        # Average window embeddings per STT segment, cluster utterance embeddings.
        seg_emb_accum: dict[int, list[np.ndarray]] = defaultdict(list)
        for pos, win_idx in enumerate(valid_indices):
            win = windows[win_idx]
            seg_emb_accum[win.stt_idx].append(X[pos])
        if len(seg_emb_accum) <= 1:
            return

        seg_ids = sorted(seg_emb_accum.keys())
        utterance_embs_raw = np.stack([np.mean(seg_emb_accum[seg_id], axis=0) for seg_id in seg_ids])
        norms = np.linalg.norm(utterance_embs_raw, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        utterance_embs = utterance_embs_raw / norms

        n_utts = len(utterance_embs)
        nn = self.config.umap_n_neighbors
        md = self.config.umap_min_dist
        dim = self.config.umap_n_components
        mcs = self.config.hdbscan_min_cluster_size

        # If min_cluster_size exceeds data size, the result will be all noise.
        if n_utts < mcs:
            print(
                f"  [Speakers] UMAP+HDBSCAN (utterance-averaged): skipped "
                f"({n_utts} utterances < min_cluster_size={mcs})"
            )
            return

        try:
            import umap as _umap
            import hdbscan as _hdbscan
        except ImportError:
            print("  [Speakers] UMAP+HDBSCAN (utterance-averaged): umap/hdbscan not installed, skipping")
            return

        # UMAP+HDBSCAN (utterance-averaged) sweep.
        # Keep it deterministic and limited to this algorithm only.
        print("  [Speakers] UMAP+HDBSCAN (utterance-averaged):")
        sweep_nns = [5, 10, 15]
        sweep_dims = [2, 5, 10]
        sweep_mcs = [5, 10, 15]

        # Always include the configured defaults first if they aren't already.
        if nn not in sweep_nns:
            sweep_nns.insert(0, nn)
        if dim not in sweep_dims:
            sweep_dims.insert(0, dim)
        if mcs not in sweep_mcs:
            sweep_mcs.insert(0, mcs)

        for nn_i in sweep_nns:
            for dim_i in sweep_dims:
                for mcs_i in sweep_mcs:
                    if n_utts < mcs_i:
                        continue
                    reducer = _umap.UMAP(
                        n_neighbors=nn_i,
                        min_dist=md,
                        n_components=dim_i,
                        metric="cosine",
                        random_state=42,
                        n_jobs=1,
                    )
                    utt_umap = reducer.fit_transform(utterance_embs)
                    clusterer = _hdbscan.HDBSCAN(
                        min_cluster_size=mcs_i,
                        min_samples=mcs_i,
                        cluster_selection_method="eom",
                    )
                    utt_labels = clusterer.fit_predict(utt_umap)
                    n_clusters = len(set(int(x) for x in utt_labels) - {-1})
                    n_noise = int(np.sum(utt_labels == -1))
                    print(
                        f"    nn={nn_i:2d} dim={dim_i:2d} mcs={mcs_i:2d} "
                        f"→ {n_clusters:3d} cluster(s), {n_noise:4d} noise"
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
                entry_id=f"utt_{idx:06d}",
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
