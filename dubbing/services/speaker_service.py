"""
speaker_service.py
──────────────────
Speaker embedding and attribution for multi-speaker audio files.

Requires:
    pip install pyannote.audio scikit-learn soundfile numpy

Default model: pyannote/wespeaker-voxceleb-resnet34-LM
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


# ─── Speaker cohesion thresholds (cosine similarity, range 0–1) ─────────────
# Bir nechta segmentlar bir xil speaker ekanligini tekshirish uchun

_COHESION_MEAN_STRICT    = 0.72   # qat'iy rejim: o'rtacha o'xshashlik
_COHESION_MEDIAN_STRICT  = 0.78   # qat'iy rejim: mediana o'xshashlik
_COHESION_P20_STRICT     = 0.58   # qat'iy rejim: 20-persentil o'xshashlik
_COHESION_MEAN_LOOSE     = 0.66   # yengil rejim: o'rtacha o'xshashlik
_COHESION_MEDIAN_LOOSE   = 0.72   # yengil rejim: mediana o'xshashlik
_COHESION_P20_LOOSE      = 0.50   # yengil rejim: 20-persentil o'xshashlik

# ─── Kam segment (≤ 2 segment) holati uchun chegara qiymatlari ───────────────

_SMALL_SAMPLE_MEDIAN_SIM = 0.58   # mediana shu darajadan yuqori → 1 speaker
_SMALL_SAMPLE_MIN_SIM    = 0.42   # minimum shu darajadan yuqori → 1 speaker

# ─── Clustering sifat chegaralari ───────────────────────────────────────────

_SILHOUETTE_WEAK_THRESHOLD    = 0.12   # bundan past → 1 speaker ehtimoli yuqori
_CENTROID_SIM_THRESHOLD       = 0.72   # 2 ta centroid juda o'xshash → 1 speaker
_CLUSTER_IMBALANCE_THRESHOLD  = 0.10   # bitta cluster juda kichik → bo'linishni rad et
_MAX_SPEAKER_CANDIDATES       = 6      # avtomatik qidirishda maksimal speaker soni

# ─── Refine (yaxshilash) bosqichi chegaralari ────────────────────────────────

_REFINE_CLOSE_MARGIN    = 0.20   # qayta belgilash uchun masofalar farqi chegarasi
_REFINE_SMOOTH_MARGIN   = 0.25   # temporal smoothing uchun keng chegara
_REFINE_VERY_SHORT_SEC  = 1.0    # bundan qisqa segmentlar har doim qayta baholanadi

# ─── Boshqa ──────────────────────────────────────────────────────────────────

_MIN_CROP_DURATION_SEC  = 0.05   # nol uzunlikli segmentdan saqlanish uchun minimal uzunlik


class SpeakerEmbeddingService:
    """
    pyannote speaker-embedding modelini yuklaydi va WAV segmentidan
    L2-normallanган fiksirlangan uzunlikdagi vektor chiqaradi.

    Standart model: pyannote/wespeaker-voxceleb-resnet34-LM
    """

    DEFAULT_MODEL = "pyannote/wespeaker-voxceleb-resnet34-LM"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        cache_dir: Optional[str] = None,
        model_dir: Optional[str] = None,
        auth_token: Optional[str] = None,
        device: Optional[str] = None,
        local_files_only: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        model_name : str
            HuggingFace model identifikatori.
        cache_dir : str, optional
            HF hub cache papkasi (HF_HUB_CACHE o'rnatadi).
        model_dir : str, optional
            Lokal model repo papkasiga to'liq yo'l.
            Misol: C:\\Users\\<ism>\\.cache\\huggingface\\hub\\models--pyannote--wespeaker-voxceleb-resnet34-LM
        auth_token : str, optional
            HuggingFace API tokeni (yopiq modellar uchun).
        device : str, optional
            Torch qurilma, masalan "cpu", "cuda", "mps".
        local_files_only : bool
            True bo'lsa HF hubga murojaat qilinmaydi (offline rejim).
        """
        self.model_name = model_name
        self.cache_dir = (str(cache_dir).strip() if cache_dir else "") or None
        self.model_dir = (str(model_dir).strip() if model_dir else "") or None
        self.auth_token = auth_token
        self.device = device
        self.local_files_only = bool(local_files_only)
        self._inference = None
        # Eslatma: os.environ __init__ da o'rnatilmaydi — model yuklanayotganda
        # lazily chaqiriladi (_configure_hf_env). Bu bir dasturda bir nechta
        # SpeakerEmbeddingService bo'lsa sozlamalarning bir-birini buzishini oldini oladi.

    # ── Ochiq API ────────────────────────────────────────────────────────────

    def get_inference(self):
        """Yuklangan pyannote Inference obyektini qaytaradi (birinchi chaqiruvda yuklanadi)."""
        return self._get_inference()

    def embed(self, wav_path: str) -> np.ndarray:
        """
        Butun WAV fayl uchun normallanган speaker embedding vektori chiqaradi.

        Parameters
        ----------
        wav_path : str
            WAV fayl yo'li.

        Returns
        -------
        np.ndarray
            1-o'lchamli float32 embedding vektori.
        """
        raw = self._get_inference()(str(wav_path))
        return self._to_vector(raw)

    # ── Ichki: model yuklash ─────────────────────────────────────────────────

    def _get_inference(self):
        """pyannote Inference obyektini lazy yuklaydi va keshda saqlaydi."""
        if self._inference is not None:
            return self._inference

        self._configure_hf_env()
        self._ensure_torchaudio_compat()

        try:
            from pyannote.audio import Inference, Model
        except ImportError as exc:
            raise RuntimeError(
                "pyannote.audio kerak. O'rnatish: pip install pyannote.audio"
            ) from exc

        model = self._load_model(Model)
        self._inference = Inference(model, window="whole", device=self.device)
        logger.info("Speaker embedding modeli yuklandi: %s", self.model_name)
        return self._inference

    def _load_model(self, Model) -> Any:
        """
        Modelni avval HF cache'dan, keyin model_dir'dan yuklashga urinadi.
        Ikkalasi ham ishlamasa aniq xato xabari ko'rsatiladi.
        """
        kwargs: Dict[str, Any] = {}
        if self.auth_token:
            kwargs["token"] = self.auth_token   # 'use_auth_token' deprecated — 'token' ishlatiladi
        if self.local_files_only:
            kwargs["local_files_only"] = True
        if self.cache_dir:
            kwargs["cache_dir"] = self.cache_dir

        primary_exc: Optional[Exception] = None

        # 1-urinish: standart HF from_pretrained
        try:
            return Model.from_pretrained(self.model_name, **kwargs)
        except Exception as exc:
            primary_exc = exc
            logger.warning(
                "%r modelini HF cache'dan yuklab bo'lmadi: %s — lokal model_dir sinab ko'riladi.",
                self.model_name,
                exc,
            )

        # 2-urinish: foydalanuvchi ko'rsatgan lokal papkadan snapshot
        model = self._try_load_from_local_model_dir(Model, kwargs)
        if model is not None:
            return model

        raise RuntimeError(
            f"Speaker embedding modeli topilmadi: {self.model_name!r}\n\n"
            "Yechim variantlari:\n"
            "  1) HF CLI orqali yuklab olish (tavsiya etiladi):\n"
            f"       huggingface-cli download {self.model_name}\n"
            "  2) SPEAKER_EMBEDDING_CACHE_DIR ni HF hub cache papkasiga o'rnating.\n"
            "  3) SPEAKER_EMBEDDING_MODEL_DIR ni model repo papkasiga o'rnating:\n"
            "       <hf_cache>/hub/models--pyannote--wespeaker-voxceleb-resnet34-LM\n\n"
            f"  Joriy model_dir={self.model_dir!r}"
        ) from primary_exc

    def _try_load_from_local_model_dir(
        self, Model, kwargs: Dict[str, Any]
    ) -> Optional[Any]:
        """model_dir ichidan snapshot topib modelni yuklashga urinadi."""
        snapshot = self._resolve_local_snapshot_path()
        if snapshot is None:
            return None

        # cache_dir argument lokal yo'lga zid keladi — olib tashlanadi
        local_kwargs = {k: v for k, v in kwargs.items() if k != "cache_dir"}
        logger.info("Lokal snapshotdan yuklanmoqda: %s", snapshot)
        try:
            return Model.from_pretrained(str(snapshot), **local_kwargs)
        except Exception as exc:
            logger.warning("Lokal snapshotdan yuklab bo'lmadi %s: %s", snapshot, exc)
            return None

    def _resolve_local_snapshot_path(self) -> Optional[Path]:
        """
        model_dir ichidan ishlatilishi mumkin bo'lgan model snapshotini topadi.
        Ikki tuzilmani qo'llab-quvvatlaydi:
          - To'g'ridan-to'g'ri model papkasi (model fayllari bevosita ichida)
          - HF hub cache tuzilmasi: <model_dir>/snapshots/<revision>/
        """
        if not self.model_dir:
            return None

        model_dir = Path(self.model_dir).expanduser().resolve()
        if not model_dir.exists():
            logger.warning("model_dir mavjud emas: %s", model_dir)
            return None

        # To'g'ridan-to'g'ri model papkasi — model fayllari bevosita ichida
        _WEIGHT_FILES = (
            "config.yaml",
            "pytorch_model.bin",
            "model.safetensors",   # yangi format (eski kodda yo'q edi)
        )
        if any((model_dir / f).exists() for f in _WEIGHT_FILES):
            return model_dir

        # HF hub cache tuzilmasi: <model_dir>/snapshots/<revision>/
        snapshots_dir = model_dir / "snapshots"
        if not snapshots_dir.exists():
            return None

        # Birinchi 'main' revision tekshiriladi
        refs_main = model_dir / "refs" / "main"
        if refs_main.exists():
            try:
                revision = refs_main.read_text(encoding="utf-8").strip()
                candidate = snapshots_dir / revision
                if candidate.exists():
                    logger.info("'main' revision snapshot ishlatilmoqda: %s", candidate)
                    return candidate
            except Exception as exc:
                logger.debug("refs/main o'qib bo'lmadi: %s", exc)

        # Eng yangi (oxirgi o'zgartirilgan) snapshotga fallback
        candidates = sorted(
            (p for p in snapshots_dir.iterdir() if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            logger.info("Eng yangi snapshot ishlatilmoqda: %s", candidates[0])
            return candidates[0]

        return None

    # ── Ichki: yordamchi metodlar ────────────────────────────────────────────

    def _configure_hf_env(self) -> None:
        """
        HuggingFace muhit o'zgaruvchilarini model yuklanishidan oldin o'rnatadi.
        __init__ emas, shu yerda chaqiriladi — muhit global holda o'zgartirilishi
        kerak bo'lganda faqat bir marta ishlaydi.
        """
        os.environ.setdefault("HF_HUB_OFFLINE", "1" if self.local_files_only else "0")
        if self.cache_dir:
            os.environ.setdefault("HF_HUB_CACHE", self.cache_dir)
            os.environ.setdefault("HUGGINGFACE_HUB_CACHE", self.cache_dir)

    def _ensure_torchaudio_compat(self) -> None:
        """Eski torchaudio versiyalarida yo'q set_audio_backend uchun shim."""
        try:
            import torchaudio  # noqa: F401
        except ImportError:
            return
        if not hasattr(torchaudio, "set_audio_backend"):
            torchaudio.set_audio_backend = lambda *args, **kwargs: None

    @staticmethod
    def _to_vector(emb: Any) -> np.ndarray:
        """Har qanday embedding (PyTorch tensor yoki ndarray) ni tekis float32 ga o'giradi."""
        if hasattr(emb, "detach"):
            emb = emb.detach().cpu().numpy()
        try:
            emb = emb.squeeze()
        except Exception as exc:
            logger.debug("Embedding squeeze qilib bo'lmadi (%s); as-is ishlatiladi.", exc)
        return np.asarray(emb, dtype="float32").reshape(-1)


class SpeakerAttributionService:
    """
    Oldindan segmentlangan transkript segmentlariga SPEAKER_00, SPEAKER_01, ...
    teglarini speaker embedding va spectral clustering yordamida yopishtiradi.

    Oddiy ishlatish
    ---------------
    emb_svc  = SpeakerEmbeddingService(model_dir="<yo'l>")
    attr_svc = SpeakerAttributionService(emb_svc)
    result   = attr_svc.assign_speakers(segments, audio_path)
    """

    def __init__(
        self,
        embedding_service: SpeakerEmbeddingService,
        auth_token: Optional[str] = None,
        default_num_speakers: Optional[int] = None,
        min_window_sec: float = 2.2,
        random_state: int = 42,
    ) -> None:
        """
        Parameters
        ----------
        embedding_service : SpeakerEmbeddingService
        auth_token : str, optional
        default_num_speakers : int, optional
            Avtomatik aniqlashni o'chirib, speaker sonini belgilash.
        min_window_sec : float
            Embedding ishonchli bo'lishi uchun minimal segment uzunligi (soniya).
            Bundan qisqa segmentlar refine bosqichida qayta baholanadi.
        random_state : int
            Takrorlanadigan clustering uchun urug'.
        """
        self.embedding_service   = embedding_service
        self.auth_token          = auth_token
        self.default_num_speakers = int(default_num_speakers) if default_num_speakers else None
        self.min_window_sec      = max(0.5, float(min_window_sec))
        self.random_state        = int(random_state)

    # ── Ochiq API ────────────────────────────────────────────────────────────

    def assign_speakers(
        self,
        segments: List[Dict],
        audio_path: str,
        num_speakers: Optional[int] = None,
    ) -> List[Dict]:
        """
        Har bir segmentga speaker tegi yopish tiradi.

        Parameters
        ----------
        segments : list of dict
            Har bir dict'da 'start' va 'end' (soniya) kalitlari bo'lishi shart.
            'text' kaliti mavjud bo'lsa natijada saqlanadi.
        audio_path : str
            To'liq audio fayl yo'li (WAV tavsiya etiladi).
        num_speakers : int, optional
            Ma'lum speaker soni; ko'rsatilmasa avtomatik aniqlanadi.

        Returns
        -------
        list of dict
            Kirish dict'lari qo'shimcha maydonlar bilan boyitilgan:
            'speaker', 'speaker_cluster', 'speaker_similarity'.
        """
        items = [dict(seg) for seg in (segments or [])]
        if not items:
            return []

        inference    = self.embedding_service.get_inference()
        embeddings   = self._extract_embeddings(items, audio_path, inference)
        speaker_count = self._resolve_num_speakers(embeddings, num_speakers, len(items))

        if speaker_count <= 1:
            labels    = np.zeros(len(items), dtype=int)
            centroids = self._compute_centroids(embeddings, labels, 1)
            return self._finalize_assignments(items, labels, embeddings, centroids)

        raw_labels    = self._cluster_embeddings(embeddings, speaker_count)
        centroids     = self._compute_centroids(embeddings, raw_labels, speaker_count)
        refined_labels = self._refine_labels(items, embeddings, raw_labels, centroids)

        # Refine'dan keyin centroidlarni qayta hisoblash (similarity aniqligi uchun)
        final_centroids = self._compute_centroids(embeddings, refined_labels, speaker_count)
        return self._finalize_assignments(items, refined_labels, embeddings, final_centroids)

    # ── Ichki: embedding chiqarish ───────────────────────────────────────────

    def _extract_embeddings(
        self,
        segments: List[Dict],
        audio_path: str,
        inference,
    ) -> np.ndarray:
        """Har bir segment uchun normallanган embedding vektori chiqaradi."""
        try:
            from pyannote.core import Segment
        except ImportError as exc:
            raise RuntimeError(
                "pyannote.core kerak. O'rnatish: pip install pyannote.audio"
            ) from exc

        audio_duration = self._get_audio_duration_sec(audio_path)
        embeddings: List[np.ndarray] = []

        for idx, seg in enumerate(segments):
            start      = float(seg.get("start", 0.0))
            end        = float(seg.get("end",   0.0))
            crop_start = max(0.0, min(start, audio_duration))
            crop_end   = max(crop_start, min(end, audio_duration))
            if crop_end <= crop_start:
                crop_end = min(audio_duration, crop_start + _MIN_CROP_DURATION_SEC)

            duration = crop_end - crop_start
            if duration < self.min_window_sec:
                logger.warning(
                    "Segment %d qisqa (%.2fs < %.2fs min_window_sec) — "
                    "embedding sifati past bo'lishi mumkin.",
                    idx, duration, self.min_window_sec,
                )

            raw     = inference.crop(audio_path, Segment(crop_start, crop_end))
            vec     = self._normalize_embedding(self.embedding_service._to_vector(raw))
            embeddings.append(vec)

            logger.debug(
                "Embedding %d [%.3f–%.3f] text=%s",
                idx, crop_start, crop_end, seg.get("text") or "",
            )

        return np.asarray(embeddings, dtype="float32")

    def _get_audio_duration_sec(self, audio_path: str) -> float:
        """Audio fayl uzunligini soniyada qaytaradi."""
        try:
            return max(0.0, float(sf.info(str(audio_path)).duration))
        except Exception as exc:
            raise RuntimeError(
                f"Audio uzunligini o'qib bo'lmadi {audio_path!r}: {exc}"
            ) from exc

    # ── Ichki: speaker soni aniqlash ─────────────────────────────────────────

    def _resolve_num_speakers(
        self,
        embeddings: np.ndarray,
        requested: Optional[int],
        segment_count: int,
    ) -> int:
        """
        Speaker sonini aniqlaydi.
        Ustuvorlik tartibi: aniq so'rov → default → avtomatik aniqlash.
        """
        if requested is not None and int(requested) > 0:
            return min(int(requested), max(1, segment_count))
        if self.default_num_speakers:
            return min(self.default_num_speakers, max(1, segment_count))
        if segment_count <= 2:
            return self._resolve_small_sample_speaker_count(embeddings, segment_count)
        return self._auto_detect_speaker_count(embeddings, segment_count)

    def _resolve_small_sample_speaker_count(
        self, embeddings: np.ndarray, segment_count: int
    ) -> int:
        """Segment soni juda kam (≤ 2) bo'lganda speaker sonini aniqlaydi."""
        if segment_count <= 1:
            return 1
        sims = self._pairwise_cosine_values(embeddings)
        if sims.size == 0:
            return 1
        median_sim = float(np.median(sims))
        min_sim    = float(np.min(sims))
        is_single  = median_sim >= _SMALL_SAMPLE_MEDIAN_SIM or min_sim >= _SMALL_SAMPLE_MIN_SIM
        logger.info(
            "Kam segment speaker soni: median_sim=%.4f min_sim=%.4f → %d",
            median_sim, min_sim, 1 if is_single else 2,
        )
        return 1 if is_single else min(2, segment_count)

    def _auto_detect_speaker_count(
        self, embeddings: np.ndarray, segment_count: int
    ) -> int:
        """
        Silhouette scoring orqali optimal speaker sonini avtomatik topadi.
        sklearn mavjud bo'lmasa embedding cohesion asosida fallback ishlatadi.
        """
        try:
            from sklearn.metrics import silhouette_score
        except ImportError:
            fallback = 1 if self._looks_like_single_speaker(embeddings) else min(4, segment_count)
            logger.warning(
                "scikit-learn topilmadi — speaker soni %d ga fallback qilindi. "
                "Aniq natija uchun: pip install scikit-learn",
                fallback,
            )
            return fallback

        if self._looks_like_single_speaker(embeddings):
            logger.info("Speaker soni = 1 (embedding cohesion tekshiruvi o'tdi).")
            return 1

        max_k = min(_MAX_SPEAKER_CANDIDATES, segment_count - 1)
        best_k, best_score = 2, None

        for k in range(2, max_k + 1):
            try:
                labels = self._cluster_embeddings(embeddings, k)
                score  = float(silhouette_score(embeddings, labels, metric="cosine"))
            except Exception as exc:
                logger.debug("k=%d uchun silhouette scoring muvaffaqiyatsiz: %s", k, exc)
                continue
            if best_score is None or score > best_score:
                best_score, best_k = score, k

        if best_score is None:
            logger.info("Silhouette scoring natija bermadi → 1 speaker.")
            return 1

        if best_score < _SILHOUETTE_WEAK_THRESHOLD:
            fallback = 1 if self._looks_like_single_speaker(embeddings, strict=False) else 2
            logger.info(
                "Zaif silhouette (%.4f < %.2f) → %d speaker.",
                best_score, _SILHOUETTE_WEAK_THRESHOLD, fallback,
            )
            return min(fallback, segment_count)

        if best_k == 2 and self._two_cluster_split_is_weak(embeddings):
            logger.info(
                "2 ta cluster ajralishi zaif (score=%.4f) → 1 speaker.", best_score
            )
            return 1

        logger.info("Speaker soni = %d (silhouette=%.4f).", best_k, best_score)
        return best_k

    # ── Ichki: bir speaker tekshiruvlari ─────────────────────────────────────

    def _looks_like_single_speaker(
        self, embeddings: np.ndarray, strict: bool = True
    ) -> bool:
        """Barcha embeddinglar bir xil speakerdan kelgan kabi ko'ringanda True qaytaradi."""
        sims = self._pairwise_cosine_values(embeddings)
        if sims.size == 0:
            return True
        mean_sim   = float(np.mean(sims))
        median_sim = float(np.median(sims))
        p20_sim    = float(np.percentile(sims, 20))
        logger.info(
            "Cohesion: mean=%.4f median=%.4f p20=%.4f strict=%s",
            mean_sim, median_sim, p20_sim, strict,
        )
        if strict:
            return (
                mean_sim   >= _COHESION_MEAN_STRICT
                and median_sim >= _COHESION_MEDIAN_STRICT
                and p20_sim    >= _COHESION_P20_STRICT
            )
        return (
            mean_sim   >= _COHESION_MEAN_LOOSE
            and median_sim >= _COHESION_MEDIAN_LOOSE
            and p20_sim    >= _COHESION_P20_LOOSE
        )

    def _two_cluster_split_is_weak(self, embeddings: np.ndarray) -> bool:
        """2 ta clusterni ajratish ma'nosiz bo'lsa True qaytaradi."""
        try:
            labels = self._cluster_embeddings(embeddings, 2)
        except Exception:
            return False

        centroids    = self._compute_centroids(embeddings, labels, 2)
        centroid_sim = float(np.dot(centroids[0], centroids[1]))
        sizes        = [int(np.sum(labels == k)) for k in (0, 1)]
        imbalance    = min(sizes) / max(1, max(sizes))
        logger.info(
            "2-cluster: centroid_sim=%.4f sizes=%s imbalance=%.4f",
            centroid_sim, sizes, imbalance,
        )
        return (
            centroid_sim >= _CENTROID_SIM_THRESHOLD
            or imbalance < _CLUSTER_IMBALANCE_THRESHOLD
        )

    # ── Ichki: clustering ────────────────────────────────────────────────────

    def _cluster_embeddings(
        self, embeddings: np.ndarray, num_speakers: int
    ) -> np.ndarray:
        """
        SpectralClustering yordamida embeddinglarni guruhlaydi.
        Barcha clustering chaqiruvlari shu bitta metod orqali o'tadi.
        """
        try:
            from sklearn.cluster import SpectralClustering
        except ImportError as exc:
            raise RuntimeError(
                "scikit-learn clustering uchun kerak. O'rnatish: pip install scikit-learn"
            ) from exc

        return SpectralClustering(
            n_clusters=int(num_speakers),
            affinity="cosine",
            assign_labels="cluster_qr",
            random_state=self.random_state,
        ).fit_predict(embeddings)

    def _compute_centroids(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        num_speakers: int,
    ) -> np.ndarray:
        """
        Har bir cluster uchun L2-normallanган o'rtacha centroid hisoblaydi.
        Bo'sh clusterlar uchun global o'rtamaga fallback qiladi.
        """
        global_mean = self._normalize_embedding(np.mean(embeddings, axis=0))
        centroids   = np.zeros((num_speakers, embeddings.shape[1]), dtype="float32")
        for k in range(num_speakers):
            mask = labels == k
            centroids[k] = (
                self._normalize_embedding(np.mean(embeddings[mask], axis=0))
                if mask.any()
                else global_mean
            )
        return centroids

    # ── Ichki: refine (yaxshilash) ───────────────────────────────────────────

    def _refine_labels(
        self,
        segments: List[Dict],
        embeddings: np.ndarray,
        raw_labels: np.ndarray,
        centroids: np.ndarray,
    ) -> np.ndarray:
        """
        Cluster teglarini davomiylik va temporal kontekst asosida yaxshilaydi.

        Ikki o'tish (punktuatsiya evristikasiz):

        1-o'tish — Juda qisqa segmentlar → eng yaqin centroidga qayta belgilash.
            Qisqa audio segmentlardan chiqqan embeddinglar ishonchsiz bo'ladi,
            shuning uchun ular to'g'ridan-to'g'ri centroid masofasi bilan
            qayta belgilanadi.

        2-o'tish — Temporal smoothing → qisqa izolyatsiyalangan segment
            bir xil qo'shnilari orasida bo'lsa va masofa farqi kichik bo'lsa,
            qo'shni speakerga belgilanadi.

        Olib tashlangan (eski kodda bor edi):
            - "!" va "?" punktuatsiya asosidagi qayta belgilash.
              Bu heuristik lingvistik asossiz va ko'p xato chiqarardi.
            - Oxirgi segmentlarda speaker_0 ga hardcode belgilash.
        """
        durations = np.array(
            [
                max(0.0, float(seg.get("end", 0.0)) - float(seg.get("start", 0.0)))
                for seg in segments
            ],
            dtype="float32",
        )
        labels = list(raw_labels)
        n      = len(labels)

        # 1-o'tish: juda qisqa segmentlarni eng yaqin centroidga qayta belgilash
        for i in range(n):
            if durations[i] < _REFINE_VERY_SHORT_SEC:
                # centroids va embeddings[i] ikkalasi ham normallanган →
                # 1 - dot_product = cosine masofasi
                distances = 1.0 - centroids.dot(embeddings[i])
                new_label = int(np.argmin(distances))
                if new_label != labels[i]:
                    logger.debug(
                        "Segment %d (%.2fs): %d → %d (eng yaqin centroid).",
                        i, float(durations[i]), labels[i], new_label,
                    )
                    labels[i] = new_label

        # 2-o'tish: temporal smoothing — izolyatsiyalangan qisqa segmentlar
        for i in range(1, n - 1):
            if durations[i] >= self.min_window_sec:
                continue
            prev_label = labels[i - 1]
            next_label = labels[i + 1]
            # Faqat: qo'shnilari bir xil, lekin joriy boshqacha bo'lsa ishlaydi
            if prev_label != next_label or labels[i] == prev_label:
                continue
            distances  = 1.0 - centroids.dot(embeddings[i])
            curr_dist  = float(distances[labels[i]])
            neigh_dist = float(distances[prev_label])
            if abs(curr_dist - neigh_dist) < _REFINE_SMOOTH_MARGIN:
                logger.debug(
                    "Segment %d (%.2fs): %d → %d (temporal smoothing).",
                    i, float(durations[i]), labels[i], prev_label,
                )
                labels[i] = prev_label

        return np.asarray(labels, dtype=int)

    # ── Ichki: yakunlash ─────────────────────────────────────────────────────

    def _finalize_assignments(
        self,
        segments: List[Dict],
        labels: np.ndarray,
        embeddings: np.ndarray,
        centroids: np.ndarray,
    ) -> List[Dict]:
        """
        Cluster raqamlarini SPEAKER_XX teglariga o'giradi va har bir segment
        uchun centroidga cosine o'xshashligini hisoblaydi.

        speaker_similarity — bu speakerning embeddingining o'z cluster
        centroidiga qanchalik yaqinligini ko'rsatadi (0–1).
        Yuqori qiymat = bu segment ushbu speaker uchun tipik.
        """
        cluster_to_speaker: Dict[int, str] = {}
        speaker_counter = 0
        result: List[Dict] = []

        for seg, label, emb in zip(segments, labels, embeddings):
            label = int(label)
            if label not in cluster_to_speaker:
                cluster_to_speaker[label] = f"SPEAKER_{speaker_counter:02d}"
                speaker_counter += 1

            speaker    = cluster_to_speaker[label]
            # emb va centroids[label] ikkalasi ham normallanган →
            # dot product = cosine similarity
            similarity = round(float(np.dot(emb, centroids[label])), 4)

            item = dict(seg)
            item["speaker"]            = speaker
            item["speaker_cluster"]    = label
            item["speaker_similarity"] = similarity
            result.append(item)

            logger.info(
                "Segment [%.3f–%.3f] %s (cluster=%d, sim=%.4f): %s",
                float(seg.get("start", 0.0)),
                float(seg.get("end",   0.0)),
                speaker,
                label,
                similarity,
                seg.get("text") or "",
            )

        logger.info("Yakuniy speaker xaritasi: %s", cluster_to_speaker)
        return result

    # ── Ichki: matematik yordamchilar ────────────────────────────────────────

    def _pairwise_cosine_values(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Barcha juftliklar uchun yuqori uchburchak cosine o'xshashlik qiymatlarini
        qaytaradi. Embeddinglar oldindan normallanmagan bo'lsa ham ishlaydi.
        """
        if embeddings is None or len(embeddings) < 2:
            return np.asarray([], dtype="float32")
        norms      = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        normalised = embeddings / norms
        matrix     = np.matmul(normalised, normalised.T)
        upper_idx  = np.triu_indices(len(normalised), k=1)
        return matrix[upper_idx].astype("float32")

    @staticmethod
    def _normalize_embedding(vector: np.ndarray) -> np.ndarray:
        """Vektorni L2-normallaydi (barqarorlik uchun kichik epsilon qo'shiladi)."""
        arr = np.asarray(vector, dtype="float32").reshape(-1)
        return arr / (np.linalg.norm(arr) + 1e-8)


# ── Modul darajasidagi yordamchi ─────────────────────────────────────────────

def cosine_similarity(a: Any, b: Any) -> float:
    """
    Ikki speaker embedding vektori o'rtasidagi cosine o'xshashligini hisoblaydi.
    PyTorch tensor yoki numpy array qabul qiladi.

    Returns
    -------
    float
        [-1, 1] oralig'ida qiymat (speaker embeddinglar uchun odatda [0, 1]).
    """
    if hasattr(a, "detach"):
        a = a.detach().cpu().numpy()
    if hasattr(b, "detach"):
        b = b.detach().cpu().numpy()
    a = np.asarray(a, dtype="float32").reshape(-1)
    b = np.asarray(b, dtype="float32").reshape(-1)
    return float(
        np.dot(a, b) / ((np.linalg.norm(a) + 1e-8) * (np.linalg.norm(b) + 1e-8))
    )