import logging
import os
from pathlib import Path

import numpy as np
import soundfile as sf


logger = logging.getLogger(__name__)


class SpeakerEmbeddingService:
    def __init__(
        self,
        model_name="pyannote/embedding",
        cache_dir=None,
        model_dir=None,
        auth_token=None,
        device=None,
        local_files_only=True,
    ):
        self.model_name = model_name
        self.cache_dir = (str(cache_dir).strip() if cache_dir else "") or None
        self.model_dir = (str(model_dir).strip() if model_dir else "") or None
        self.auth_token = auth_token
        self.device = device
        self.local_files_only = bool(local_files_only)
        self._inference = None

        os.environ.setdefault("HF_HUB_OFFLINE", "1" if self.local_files_only else "0")
        if self.cache_dir:
            os.environ.setdefault("HF_HUB_CACHE", self.cache_dir)
            os.environ.setdefault("HUGGINGFACE_HUB_CACHE", self.cache_dir)

    def get_inference(self):
        return self._get_inference()

    def embed(self, wav_path):
        inference = self._get_inference()
        emb = inference(str(wav_path))
        return self._to_vector(emb)

    def _to_vector(self, emb):
        if hasattr(emb, "detach"):
            emb = emb.detach().cpu().numpy()
        try:
            emb = emb.squeeze()
        except Exception:
            pass
        return np.asarray(emb, dtype="float32").reshape(-1)

    def _get_inference(self):
        if self._inference is not None:
            return self._inference

        self._ensure_torchaudio_compat()

        try:
            from pyannote.audio import Inference, Model
        except ImportError as exc:
            raise RuntimeError(
                "pyannote.audio is required for speaker embeddings. Install with: pip install pyannote.audio"
            ) from exc

        kwargs = {}
        if self.auth_token:
            kwargs["use_auth_token"] = self.auth_token

        model_ref = self.model_name
        try:
            if self.local_files_only:
                kwargs["local_files_only"] = True
            if self.cache_dir:
                kwargs["cache_dir"] = self.cache_dir
            model = Model.from_pretrained(model_ref, **kwargs)
        except Exception as exc:
            model = self._try_load_from_local_model_dir(Model, kwargs)
            if model is None:
                hint = (
                    "Speaker embedding model is not available locally.\n"
                    "Fix options:\n"
                    "1) Download it into the default HF cache (recommended), then rerun.\n"
                    "2) Or set SPEAKER_EMBEDDING_CACHE_DIR to your HF hub cache directory and rerun.\n"
                    "3) Or set SPEAKER_EMBEDDING_MODEL_DIR to a HF cached repo directory\n"
                    "   like ...\\hub\\models--pyannote--embedding so a local snapshot can be used.\n"
                    "\n"
                    "Example:\n"
                    "  huggingface-cli login\n"
                    "  huggingface-cli download pyannote/embedding\n"
                    "  # Optional if your cache is non-default:\n"
                    "  # SPEAKER_EMBEDDING_CACHE_DIR=C:\\Users\\AKBAR\\.cache\\huggingface\\hub\n"
                    "  # Or point directly at the cached repo directory:\n"
                    "  # SPEAKER_EMBEDDING_MODEL_DIR=C:\\Users\\AKBAR\\.cache\\huggingface\\hub\\models--pyannote--embedding\n"
                    "\n"
                    f"Details: model_ref={model_ref!r} model_dir={self.model_dir!r} error={exc!s}\n"
                )
                raise RuntimeError(hint) from exc

        self._inference = Inference(model, window="whole", device=self.device)
        return self._inference

    def _try_load_from_local_model_dir(self, Model, kwargs):
        snapshot_path = self._resolve_local_snapshot_path()
        if not snapshot_path:
            return None

        local_kwargs = dict(kwargs)
        local_kwargs.pop("cache_dir", None)
        logger.info("Trying speaker embedding model from local snapshot: %s", snapshot_path)
        try:
            return Model.from_pretrained(str(snapshot_path), **local_kwargs)
        except Exception as exc:
            logger.warning("Failed to load speaker embedding from local snapshot %s: %s", snapshot_path, exc)
            return None

    def _resolve_local_snapshot_path(self):
        if not self.model_dir:
            return None

        model_dir = Path(self.model_dir).expanduser().resolve()
        if not model_dir.exists():
            return None

        if (model_dir / "config.yaml").exists() or (model_dir / "pytorch_model.bin").exists():
            return model_dir

        snapshots_dir = model_dir / "snapshots"
        if not snapshots_dir.exists():
            return None

        refs_main = model_dir / "refs" / "main"
        if refs_main.exists():
            try:
                revision = refs_main.read_text(encoding="utf-8").strip()
                candidate = snapshots_dir / revision
                if candidate.exists():
                    return candidate
            except Exception:
                pass

        candidates = [path for path in snapshots_dir.iterdir() if path.is_dir()]
        if not candidates:
            return None
        candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        return candidates[0]

    def _ensure_torchaudio_compat(self):
        try:
            import torchaudio
        except Exception:
            return

        if not hasattr(torchaudio, "set_audio_backend"):
            torchaudio.set_audio_backend = lambda *args, **kwargs: None


class SpeakerAttributionService:
    def __init__(
        self,
        embedding_service,
        auth_token=None,
        default_num_speakers=None,
        min_window_sec=2.2,
        random_state=42,
    ):
        self.embedding_service = embedding_service
        self.auth_token = auth_token
        self.default_num_speakers = int(default_num_speakers) if default_num_speakers else None
        self.min_window_sec = max(0.5, float(min_window_sec))
        self.random_state = int(random_state)

    def assign_speakers(self, segments, audio_path, num_speakers=None):
        items = [dict(segment) for segment in (segments or [])]
        if not items:
            return []

        inference = self.embedding_service.get_inference()
        embeddings = self._extract_embeddings(items, audio_path, inference)
        speaker_count = self._resolve_num_speakers(embeddings, num_speakers, len(items))
        if speaker_count <= 1:
            return self._finalize_assignments(items, np.zeros(len(items), dtype=int))

        raw_labels = self._cluster_embeddings(embeddings, speaker_count)
        refined_labels = self._refine_labels(items, embeddings, raw_labels, speaker_count)
        return self._finalize_assignments(items, refined_labels)

    def _extract_embeddings(self, segments, audio_path, inference):
        try:
            from pyannote.core import Segment
        except ImportError as exc:
            raise RuntimeError(
                "pyannote.core is required for speaker attribution. Install with: pip install pyannote.core"
            ) from exc

        audio_duration = self._get_audio_duration_sec(audio_path)
        embeddings = []
        for idx, seg in enumerate(segments):
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", 0.0))
            crop_start = max(0.0, min(start, audio_duration))
            crop_end = max(crop_start, min(end, audio_duration))
            if crop_end <= crop_start:
                crop_end = min(audio_duration, crop_start + 0.05)

            sub_segment = Segment(crop_start, crop_end)
            emb = inference.crop(audio_path, sub_segment)
            vector = self._normalize_embedding(self.embedding_service._to_vector(emb))
            embeddings.append(vector)
            logger.info(
                "Speaker embedding segment %d start=%.3f end=%.3f crop_start=%.3f crop_end=%.3f text=%s",
                idx,
                start,
                end,
                crop_start,
                crop_end,
                seg.get("text") or "",
            )
        return np.asarray(embeddings, dtype="float32")

    def _get_audio_duration_sec(self, audio_path):
        try:
            info = sf.info(str(audio_path))
            return max(0.0, float(info.duration))
        except Exception as exc:
            raise RuntimeError(f"Failed to read speaker audio duration: {exc}") from exc

    def _resolve_num_speakers(self, embeddings, requested_num_speakers, segment_count):
        if requested_num_speakers is not None and int(requested_num_speakers) > 0:
            return min(int(requested_num_speakers), max(1, int(segment_count)))
        if self.default_num_speakers:
            return min(int(self.default_num_speakers), max(1, int(segment_count)))
        if segment_count <= 2:
            return self._resolve_small_sample_speaker_count(embeddings, segment_count)

        try:
            from sklearn.cluster import SpectralClustering
            from sklearn.metrics import silhouette_score
        except ImportError:
            return 1 if self._looks_like_single_speaker(embeddings) else min(4, segment_count)

        if self._looks_like_single_speaker(embeddings):
            logger.info("Speaker count resolved as 1 by embedding cohesion")
            return 1

        max_candidates = min(6, segment_count - 1)
        best_k = 2
        best_score = None
        for k in range(2, max_candidates + 1):
            try:
                labels = SpectralClustering(
                    n_clusters=k,
                    affinity="cosine",
                    assign_labels="cluster_qr",
                    random_state=self.random_state,
                ).fit_predict(embeddings)
                score = silhouette_score(embeddings, labels, metric="cosine")
            except Exception:
                continue
            if best_score is None or score > best_score:
                best_score = score
                best_k = k

        if best_score is None:
            logger.info(
                "Speaker count resolved as 1 because clustering produced no usable score: best_score=%s",
                best_score,
            )
            return 1
        if best_score < 0.12:
            fallback = 1 if self._looks_like_single_speaker(embeddings, strict=False) else 2
            logger.info(
                "Speaker count resolved as %d by weak clustering evidence: best_score=%.4f",
                fallback,
                float(best_score),
            )
            return min(fallback, segment_count)

        if best_k == 2 and self._two_cluster_split_is_weak(embeddings):
            logger.info(
                "Speaker count resolved as 1 by weak two-cluster separation: best_score=%.4f",
                float(best_score),
            )
            return 1

        logger.info("Speaker count resolved as %d by silhouette score %.4f", best_k, float(best_score))
        return best_k

    def _resolve_small_sample_speaker_count(self, embeddings, segment_count):
        if segment_count <= 1:
            return 1
        similarity = self._pairwise_cosine_values(embeddings)
        if similarity.size == 0:
            return 1
        median_sim = float(np.median(similarity))
        min_sim = float(np.min(similarity))
        if median_sim >= 0.58 or min_sim >= 0.42:
            logger.info(
                "Speaker count resolved as 1 for small sample: median_sim=%.4f min_sim=%.4f",
                median_sim,
                min_sim,
            )
            return 1
        return min(2, segment_count)

    def _looks_like_single_speaker(self, embeddings, strict=True):
        similarity = self._pairwise_cosine_values(embeddings)
        if similarity.size == 0:
            return True
        median_sim = float(np.median(similarity))
        p20_sim = float(np.percentile(similarity, 20))
        mean_sim = float(np.mean(similarity))
        min_sim = float(np.min(similarity))
        logger.info(
            "Speaker cohesion stats: mean_sim=%.4f median_sim=%.4f p20_sim=%.4f min_sim=%.4f strict=%s",
            mean_sim,
            median_sim,
            p20_sim,
            min_sim,
            strict,
        )
        if strict:
            return mean_sim >= 0.72 and median_sim >= 0.78 and p20_sim >= 0.58
        return mean_sim >= 0.66 and median_sim >= 0.72 and p20_sim >= 0.50

    def _two_cluster_split_is_weak(self, embeddings):
        try:
            labels = self._cluster_embeddings(embeddings, 2)
        except Exception:
            return False

        values = []
        for cluster_id in (0, 1):
            points = embeddings[labels == cluster_id]
            if len(points) == 0:
                return True
            centroid = self._normalize_embedding(np.mean(points, axis=0))
            values.append(centroid)

        centroid_sim = float(np.dot(values[0], values[1]))
        sizes = [int(np.sum(labels == cluster_id)) for cluster_id in (0, 1)]
        imbalance = min(sizes) / max(1, max(sizes))
        logger.info(
            "Two-cluster separation stats: centroid_sim=%.4f sizes=%s imbalance=%.4f",
            centroid_sim,
            sizes,
            imbalance,
        )
        return centroid_sim >= 0.72 or imbalance < 0.10

    def _pairwise_cosine_values(self, embeddings):
        if embeddings is None or len(embeddings) < 2:
            return np.asarray([], dtype="float32")
        normalized = np.asarray([self._normalize_embedding(item) for item in embeddings], dtype="float32")
        matrix = np.matmul(normalized, normalized.T)
        upper = np.triu_indices(len(normalized), k=1)
        return matrix[upper].astype("float32")

    def _cluster_embeddings(self, embeddings, num_speakers):
        try:
            from sklearn.cluster import SpectralClustering
        except ImportError as exc:
            raise RuntimeError(
                "scikit-learn is required for speaker clustering. Install with: pip install scikit-learn"
            ) from exc

        return SpectralClustering(
            n_clusters=int(num_speakers),
            affinity="cosine",
            assign_labels="cluster_qr",
            random_state=self.random_state,
        ).fit_predict(embeddings)

    def _refine_labels(self, segments, embeddings, raw_labels, num_speakers):
        try:
            from scipy.spatial.distance import cdist
        except ImportError as exc:
            raise RuntimeError(
                "scipy is required for speaker refinement. Install with: pip install scipy"
            ) from exc

        durations = np.asarray(
            [max(0.0, float(seg.get("end", 0.0)) - float(seg.get("start", 0.0))) for seg in segments],
            dtype="float32",
        )
        refined_labels = list(raw_labels)
        centroids = np.zeros((int(num_speakers), embeddings.shape[1]), dtype="float32")
        for cluster_id in range(int(num_speakers)):
            points = embeddings[raw_labels == cluster_id]
            if len(points) > 0:
                centroids[cluster_id] = np.mean(points, axis=0)
            else:
                centroids[cluster_id] = np.mean(embeddings, axis=0)
            centroids[cluster_id] = self._normalize_embedding(centroids[cluster_id])

        for i in range(1, len(segments)):
            curr_text = (segments[i].get("text") or "").strip()
            prev_text = (segments[i - 1].get("text") or "").strip()

            distances = cdist(embeddings[i].reshape(1, -1), centroids, metric="cosine").flatten()
            closest_speakers = np.argsort(distances)

            if durations[i] < 2.2 and refined_labels[i] == refined_labels[i - 1]:
                if abs(float(distances[closest_speakers[0]]) - float(distances[closest_speakers[1]])) < 0.20:
                    if curr_text.endswith("!") or curr_text.endswith("?"):
                        refined_labels[i] = int(closest_speakers[1])

            if refined_labels[i] == refined_labels[i - 1] and (prev_text.endswith("!") or prev_text.endswith("?")):
                if abs(float(distances[closest_speakers[0]]) - float(distances[closest_speakers[1]])) < 0.25:
                    refined_labels[i] = int(closest_speakers[1])

        for i in range(len(segments) - 2, -1, -1):
            curr_text = (segments[i].get("text") or "").strip()
            if durations[i] < 1.5 and curr_text.endswith("!") and refined_labels[i + 1] == 0:
                distances = cdist(embeddings[i].reshape(1, -1), centroids, metric="cosine").flatten()
                order = np.argsort(distances)
                if float(distances[0]) < 0.35 or (len(order) > 1 and int(order[1]) == 0):
                    refined_labels[i] = 0

        return np.asarray(refined_labels, dtype=int)

    def _finalize_assignments(self, segments, labels):
        cluster_to_speaker = {}
        speaker_counter = 0
        final_dialogue = []
        for idx, (seg, label) in enumerate(zip(segments, labels)):
            label = int(label)
            if label not in cluster_to_speaker:
                cluster_to_speaker[label] = f"SPEAKER_{speaker_counter:02d}"
                speaker_counter += 1
            speaker = cluster_to_speaker[label]
            logger.info(
                "Sentence unit start=%.3f end=%.3f speaker=%s cluster=%d text=%s",
                float(seg.get("start", 0.0)),
                float(seg.get("end", 0.0)),
                speaker,
                label,
                seg.get("text") or "",
            )
            item = dict(seg)
            item["speaker"] = speaker
            item["speaker_cluster"] = label
            item["speaker_similarity"] = None
            final_dialogue.append(item)
        logger.info("Embedding-based sentence speaker assignment: %s", cluster_to_speaker)
        return final_dialogue

    def _normalize_embedding(self, vector):
        arr = np.asarray(vector, dtype="float32").reshape(-1)
        norm = np.linalg.norm(arr) + 1e-8
        return arr / norm


def cosine_similarity(a, b):
    if hasattr(a, "detach"):
        a = a.detach().cpu().numpy()
    if hasattr(b, "detach"):
        b = b.detach().cpu().numpy()

    a = np.asarray(a, dtype="float32").reshape(-1)
    b = np.asarray(b, dtype="float32").reshape(-1)
    an = np.linalg.norm(a) + 1e-8
    bn = np.linalg.norm(b) + 1e-8
    return float((a @ b) / (an * bn))
