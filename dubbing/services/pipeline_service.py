import logging
import re
from dataclasses import dataclass
from pathlib import Path


from .ffmpeg_service import FFmpegService
from .prosody_service import ProsodyDurationMapperService
from .speaker_service import SpeakerEmbeddingService, cosine_similarity
from .translation_service import LocalNLLBTranslationService
from .tts_service import CoquiTTSService
from .whisper_service import WhisperService


logger = logging.getLogger(__name__)


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass
class _RawSegment:
    start: float
    end: float
    text: str


class DubbingPipelineService:
    """
    Pipeline overview (current behavior):
    1) Extract audio from video -> 16kHz mono WAV
    2) Whisper (Groq) transcription with timestamps (verbose_json segments)
    3) Split Whisper segments into sentence-like subsegments with proportional timestamps
    4) For each subsegment:
       - extract its audio to wav
       - compute speaker embedding
       - assign speaker_id by cosine similarity clustering
    5) Merge adjacent subsegments belonging to the same speaker within MAX_SPEAKER_GAP_SECONDS
    6) Translate all merged segments in a single local NLLB batch
    7) Build per-speaker reference wav (concat segments) targeting ~8s total duration
    8) XTTS-v2: synthesize each segment using that speaker reference, map duration, mix on timeline, mux into video
    """

    def __init__(
        self,
        ffmpeg_bin="ffmpeg",
        whisper_model_name="whisper-large-v3",
        groq_api_key="",
        translation_model_dir="",
        nllb_source_lang="eng_Latn",
        nllb_target_lang="kaz_Cyrl",
        nllb_batch_size=8,
        nllb_max_new_tokens=256,
        source_language_name="English",
        target_language_name="Kazakh",
        source_language=None,
        diarization_model_name=None,  # kept for backward compatibility; unused in similarity pipeline
        hf_auth_token=None,
        diarization_min_speakers=None,  # kept for backward compatibility; unused
        diarization_max_speakers=None,  # kept for backward compatibility; unused
        tts_xtts_local_dir=None,
        xtts_fallback_language="tr",
        tts_target_language="",
        # The following knobs default to sane values and are intended to be auto-tuned.
        # We keep them as parameters for code-level control, but do not require .env tuning.
        min_segment_duration=0.4,
        max_speaker_gap_seconds=0.2,
        min_tempo_factor=0.90,
        max_tempo_factor=1.60,
        speaker_ref_target_seconds=8.0,
        speaker_ref_min_seconds=6.0,
        speaker_ref_max_seconds=12.0,
        speaker_ref_min_segment_seconds=0.75,
        speaker_embedding_model_name="pyannote/embedding",
        speaker_embedding_cache_dir=None,
        speaker_embedding_min_seconds=2.0,
        speaker_embedding_device=None,
        ref_audio_output_dir=None,
    ):
        self.ffmpeg = FFmpegService(ffmpeg_bin=ffmpeg_bin)
        self.whisper = WhisperService(model_name=whisper_model_name, api_key=groq_api_key)
        self.translation = LocalNLLBTranslationService(
            model_dir=translation_model_dir,
            source_lang=nllb_source_lang,
            target_lang=nllb_target_lang,
            batch_size=nllb_batch_size,
            max_new_tokens=nllb_max_new_tokens,
        )
        self.tts = CoquiTTSService(
            xtts_local_dir=tts_xtts_local_dir,
            xtts_fallback_language=xtts_fallback_language,
        )
        self.prosody = ProsodyDurationMapperService(
            ffmpeg_service=self.ffmpeg,
            min_tempo_factor=min_tempo_factor,
            max_tempo_factor=max_tempo_factor,
        )

        self.source_language_name = source_language_name
        self.target_language_name = target_language_name
        self.source_language = source_language
        self.tts_target_language = (tts_target_language or "").strip().lower()

        self.min_segment_duration = float(min_segment_duration)
        self.max_speaker_gap_seconds = float(max_speaker_gap_seconds)

        self.speaker_ref_target_seconds = float(speaker_ref_target_seconds)
        self.speaker_ref_min_seconds = float(speaker_ref_min_seconds)
        self.speaker_ref_max_seconds = float(speaker_ref_max_seconds)
        self.speaker_ref_min_segment_seconds = float(speaker_ref_min_segment_seconds)

        self.speaker_embedding_min_seconds = float(speaker_embedding_min_seconds)
        self.speaker_embedder = SpeakerEmbeddingService(
            model_name=speaker_embedding_model_name,
            cache_dir=speaker_embedding_cache_dir,
            auth_token=hf_auth_token,
            device=speaker_embedding_device,
        )
        self.ref_audio_output_dir = (
            str(ref_audio_output_dir).strip() if ref_audio_output_dir else ""
        ) or None

    def run(self, input_video_path, extracted_audio_path, tts_audio_path, output_video_path):
        # 1) Extract full audio
        self.ffmpeg.extract_audio(input_video_path=input_video_path, output_audio_path=extracted_audio_path)
        total_duration = self.ffmpeg.get_duration(extracted_audio_path)

        # 2) ASR
        asr = self.whisper.transcribe(extracted_audio_path, language=self.source_language)
        detected_language = asr.get("language")
        whisper_segments = asr.get("segments") or []

        if not whisper_segments:
            raise RuntimeError("Whisper returned no segments")

        # 3) Split into sentence-like subsegments
        raw_segments = self._split_whisper_segments_into_subsegments(whisper_segments)
        raw_segments = self._merge_too_short_segments(raw_segments, min_dur=self.min_segment_duration)

        # 4) Extract audio per segment + embedding + speaker assignment
        work_dir = Path(extracted_audio_path).resolve().parent
        segment_audio_paths = []
        embeddings = []
        for i, seg in enumerate(raw_segments):
            # For embeddings, prefer a slightly longer window than the exact segment to stabilize similarity.
            emb_start, emb_end = self._expand_window(
                seg.start, seg.end, min_seconds=self.speaker_embedding_min_seconds, total_duration=total_duration
            )
            wav_path = work_dir / f"seg_{i:04d}_emb.wav"
            self.ffmpeg.extract_audio_segment(
                input_audio_path=extracted_audio_path,
                output_audio_path=str(wav_path),
                start_sec=emb_start,
                end_sec=emb_end,
            )
            denoised_path = work_dir / f"seg_{i:04d}_emb_denoised.wav"
            self.ffmpeg.denoise_audio(
                input_audio_path=str(wav_path),
                output_audio_path=str(denoised_path),
            )
            segment_audio_paths.append(str(denoised_path))
            embeddings.append(self.speaker_embedder.embed(denoised_path))
            logger.info(
                "Embedding window %d: %.2f-%.2f (orig %.2f-%.2f) denoised=%s text=%s",
                i,
                emb_start,
                emb_end,
                seg.start,
                seg.end,
                denoised_path.name,
                (seg.text or "").strip(),
            )

        speaker_ids = self._assign_speakers_offline(embeddings, raw_segments)

        # 5) Merge adjacent segments by speaker and gap
        merged = self._merge_adjacent_by_speaker(raw_segments, speaker_ids, max_gap=self.max_speaker_gap_seconds)

        # 6) Translate in one local batch
        src_texts = [m["text"] for m in merged]
        translations = self.translation.translate_batch(src_texts)

        # 7) Build speaker reference wavs (target ~8s)
        speaker_to_seg_audio = self._collect_segment_audio_by_speaker(
            merged=merged,
            extracted_audio_path=extracted_audio_path,
            work_dir=work_dir,
        )
        speaker_refs = self._build_speaker_references(speaker_to_seg_audio, work_dir=work_dir)
        self._persist_reference_audio(speaker_refs)

        # Log speaker refs
        for spk, info in sorted(speaker_refs.items(), key=lambda kv: kv[0]):
            logger.info(
                "Speaker %s reference built: %.2fs -> %s",
                spk,
                info["duration_sec"],
                info["path"],
            )

        # 8) TTS each segment -> duration map -> mix timeline
        timeline_segments = []
        transcript_text = []
        translated_text = []

        timeline_shift = 0.0
        for idx, (m, tr_text) in enumerate(zip(merged, translations)):
            spk = m["speaker"]
            start = float(m["start"])
            end = float(m["end"])
            # Each segment must occupy the window [start, next_start) without cutting audio.
            # If there is a gap, we fill it (slow-down or pad); if it doesn't fit, we speed-up.
            if idx + 1 < len(merged):
                next_start = float(merged[idx + 1]["start"])
                window_end = next_start if next_start > start else end
            else:
                window_end = end
            base_window = max(0.01, window_end - start)

            ref_path = None
            if spk in speaker_refs:
                ref_path = speaker_refs[spk]["path"]
            if not ref_path:
                # fallback: use the segment itself as reference (still better than None)
                ref_path = m["audio_path"]

            logger.info(
                "Segment %d [%s %.2f-%.2f] Whisper: %s",
                idx,
                spk,
                start,
                end,
                m["text"],
            )
            logger.info(
                "Segment %d [%s %.2f-%.2f] Translation: %s",
                idx,
                spk,
                start,
                end,
                tr_text,
            )
            logger.info(
                "Segment %d [%s %.2f-%.2f] SpeakerRef: %s",
                idx,
                spk,
                start,
                end,
                ref_path,
            )

            transcript_text.append(m["text"])
            translated_text.append(tr_text)

            seg_tts_path = str(work_dir / f"tts_{idx:04d}.wav")
            seg_mapped_path = str(work_dir / f"tts_{idx:04d}_mapped.wav")

            self.tts.synthesize_to_file(
                text=tr_text,
                output_audio_path=seg_tts_path,
                speaker_reference_audio=ref_path,
                language=self.tts_target_language,
            )
            source_duration = self.ffmpeg.get_duration(seg_tts_path)
            target_duration = base_window
            max_speedup = float(self.prosody.max_tempo_factor)
            needed_factor = source_duration / max(0.01, base_window)
            if needed_factor > max_speedup:
                # Avoid too-fast speech: keep tempo at max, extend the window instead.
                target_duration = source_duration / max_speedup
                overflow = target_duration - base_window
                timeline_shift += max(0.0, overflow)
                logger.warning(
                    "Segment %d exceeds max tempo. source=%.2fs window=%.2fs needed=%.2f max=%.2f -> extend by %.2fs",
                    idx,
                    source_duration,
                    base_window,
                    needed_factor,
                    max_speedup,
                    max(0.0, overflow),
                )

            self.prosody.map_to_duration(
                input_audio_path=seg_tts_path,
                output_audio_path=seg_mapped_path,
                target_duration_sec=target_duration,
            )

            timeline_segments.append({"path": seg_mapped_path, "start": start + timeline_shift})

        total_duration = max(total_duration, (merged[-1]["end"] + timeline_shift)) if merged else total_duration
        self.ffmpeg.mix_segments_on_timeline(
            segments=timeline_segments,
            output_audio_path=tts_audio_path,
            total_duration_sec=total_duration,
        )
        self.ffmpeg.mux_audio_with_video(
            input_video_path=input_video_path,
            input_audio_path=tts_audio_path,
            output_video_path=output_video_path,
        )

        return {
            "detected_language": detected_language,
            "segments_count": len(merged),
            "transcript_text": " ".join([t for t in transcript_text if t]).strip(),
            "translated_text": " ".join([t for t in translated_text if t]).strip(),
            "speaker_references": {k: v["path"] for k, v in speaker_refs.items()},
        }

    def _persist_reference_audio(self, speaker_refs):
        if not self.ref_audio_output_dir or not speaker_refs:
            return

        out_dir = Path(self.ref_audio_output_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        for spk, info in speaker_refs.items():
            src = Path(info["path"])
            if not src.exists():
                continue
            dest = out_dir / src.name
            try:
                dest.write_bytes(src.read_bytes())
                logger.info("Speaker %s reference copied to: %s", spk, dest)
            except Exception as exc:
                logger.warning("Failed to persist speaker ref %s to %s: %s", spk, dest, exc)

    def _split_whisper_segments_into_subsegments(self, whisper_segments):
        out = []
        for seg in whisper_segments:
            try:
                start = float(seg["start"])
                end = float(seg["end"])
            except Exception:
                continue
            text = (seg.get("text") or "").strip()
            if not text or end <= start:
                continue

            parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(text) if p.strip()]
            if len(parts) <= 1:
                out.append(_RawSegment(start=start, end=end, text=text))
                continue

            total_chars = sum(max(1, len(p)) for p in parts)
            cursor = start
            total_dur = max(0.01, end - start)
            for i, p in enumerate(parts):
                frac = max(1, len(p)) / total_chars
                dur = total_dur * frac
                if i == len(parts) - 1:
                    seg_end = end
                else:
                    seg_end = min(end, cursor + dur)
                if seg_end <= cursor:
                    continue
                out.append(_RawSegment(start=cursor, end=seg_end, text=p))
                cursor = seg_end

        return sorted(out, key=lambda s: (s.start, s.end))

    def _merge_too_short_segments(self, segments, min_dur):
        min_dur = float(min_dur)
        if min_dur <= 0:
            return segments

        merged = []
        buf = None
        for s in segments:
            if buf is None:
                buf = _RawSegment(start=s.start, end=s.end, text=s.text)
                continue
            if (buf.end - buf.start) >= min_dur:
                merged.append(buf)
                buf = _RawSegment(start=s.start, end=s.end, text=s.text)
                continue

            # Merge with next to reach min duration
            buf = _RawSegment(start=buf.start, end=s.end, text=(buf.text + " " + s.text).strip())

        if buf is not None:
            merged.append(buf)

        # Drop any pathological zero-length entries
        merged = [m for m in merged if (m.end - m.start) > 0.01 and (m.text or "").strip()]
        return merged

    def _assign_speakers_offline(self, embeddings, segments):
        """
        Assign speaker ids using global clustering on embeddings (cosine distance).
        This avoids "speaker-id explosion" on short segments and usually matches dialogue patterns better.
        """
        if not embeddings:
            return []

        # Small n: trivial case.
        if len(embeddings) == 1:
            return ["SPEAKER_00"]

        labels, meta = self._cluster_embeddings_auto_k(embeddings)
        if meta:
            logger.info(
                "Speaker clustering: chosen_k=%s candidates=%s best_silhouette=%.4f",
                meta.get("chosen_k"),
                meta.get("candidates"),
                meta.get("best_score", -1.0),
            )
        labels = self._temporal_smooth_labels(labels)
        labels = self._reassign_tiny_clusters(labels, embeddings, segments)

        # Map cluster labels to SPEAKER_XX by order of first appearance in time.
        order = []
        for lab in labels:
            if lab not in order:
                order.append(lab)
        mapping = {lab: f"SPEAKER_{i:02d}" for i, lab in enumerate(order)}

        # Log per-cluster duration totals.
        totals = {}
        for lab, seg in zip(labels, segments):
            totals[lab] = totals.get(lab, 0.0) + max(0.01, float(seg.end) - float(seg.start))
        for lab in order:
            logger.info("Cluster %s duration: %.2fs", lab, totals.get(lab, 0.0))

        # Log per-segment assignment.
        for idx, (lab, seg) in enumerate(zip(labels, segments)):
            logger.info(
                "Segment %d cluster=%s start=%.2f end=%.2f text=%s",
                idx,
                lab,
                seg.start,
                seg.end,
                (seg.text or "").strip(),
            )

        return [mapping[lab] for lab in labels]

    def _merge_adjacent_by_speaker(self, segments, speaker_ids, max_gap):
        max_gap = float(max_gap)
        merged = []
        prev = None
        prev_spk = None

        for s, spk in zip(segments, speaker_ids):
            if prev is None:
                prev = {"start": s.start, "end": s.end, "text": s.text, "speaker": spk}
                prev_spk = spk
                continue

            gap = float(s.start) - float(prev["end"])
            if spk == prev_spk and gap <= max_gap:
                prev["end"] = s.end
                prev["text"] = (prev["text"] + " " + s.text).strip()
                continue

            merged.append(prev)
            prev = {"start": s.start, "end": s.end, "text": s.text, "speaker": spk}
            prev_spk = spk

        if prev is not None:
            merged.append(prev)

        return merged

    def _collect_segment_audio_by_speaker(self, merged, extracted_audio_path, work_dir):
        speaker_to_paths = {}
        for idx, m in enumerate(merged):
            start = float(m["start"])
            end = float(m["end"])
            spk = m["speaker"]
            dur = max(0.0, end - start)

            wav_path = Path(work_dir) / f"merged_{idx:04d}_{spk}.wav"
            self.ffmpeg.extract_audio_segment(
                input_audio_path=extracted_audio_path,
                output_audio_path=str(wav_path),
                start_sec=start,
                end_sec=end,
            )
            m["audio_path"] = str(wav_path)
            m["duration_sec"] = dur
            speaker_to_paths.setdefault(spk, []).append(str(wav_path))

        return speaker_to_paths

    def _build_speaker_references(self, speaker_to_paths, work_dir):
        refs = {}
        for spk, paths in speaker_to_paths.items():
            # Prefer longer segments, but allow concatenating many short segments to reach ~8 seconds.
            candidates = []
            for p in paths:
                try:
                    d = self.ffmpeg.get_duration(p)
                except Exception:
                    continue
                if d <= 0.05:
                    continue
                # Keep extremely short segments out, but don't require long ones.
                if d < min(0.25, self.speaker_ref_min_segment_seconds):
                    continue
                candidates.append((p, d))

            if not candidates:
                logger.warning("Speaker %s reference: no candidates after filtering.", spk)
                continue

            # Prefer longer segments, then greedily approach target duration within [min, max].
            candidates.sort(key=lambda x: x[1], reverse=True)
            chosen = []
            total = 0.0
            target = self.speaker_ref_target_seconds
            min_total = self.speaker_ref_min_seconds
            max_total = self.speaker_ref_max_seconds

            for p, d in candidates:
                if total >= target and total >= min_total:
                    break

                if total + d > max_total:
                    continue

                # Add if it improves distance to target, or if we still need to reach min_total.
                dist_before = abs(target - total)
                dist_after = abs(target - (total + d))
                if total < min_total or dist_after <= dist_before:
                    chosen.append(p)
                    total += d

            # Ensure we meet min_total if possible (even if it overshoots target slightly).
            if total < min_total:
                for p, d in candidates:
                    if p in chosen:
                        continue
                    if total + d > max_total:
                        continue
                    chosen.append(p)
                    total += d
                    if total >= min_total:
                        break

            if total < min_total:
                # As a last resort, still build something to avoid crashing XTTS.
                chosen = [candidates[0][0]]
                total = candidates[0][1]

            out_path = str(Path(work_dir) / f"ref_{spk}.wav")
            self.ffmpeg.concat_audio_files(chosen, out_path)
            denoised_path = str(Path(work_dir) / f"ref_{spk}_denoised.wav")
            try:
                self.ffmpeg.denoise_audio(out_path, denoised_path)
                out_path = denoised_path
            except Exception as exc:
                logger.warning("Failed to denoise speaker ref %s: %s", spk, exc)
            logger.info(
                "Speaker %s reference details: target=%.2fs min=%.2fs max=%.2fs chosen_total=%.2fs chosen_files=%s",
                spk,
                target,
                min_total,
                max_total,
                total,
                [Path(p).name for p in chosen],
            )
            refs[spk] = {"path": out_path, "duration_sec": total}

        return refs

    def _cluster_embeddings_auto_k(self, embeddings):
        import numpy as np

        X = np.stack([np.asarray(e, dtype="float32").reshape(-1) for e in embeddings], axis=0)
        n = X.shape[0]
        meta = {"chosen_k": 1, "candidates": [], "best_score": -1.0}

        # If clustering deps aren't available, fall back to "everyone is the same speaker".
        try:
            from sklearn.cluster import AgglomerativeClustering
            from sklearn.metrics import silhouette_score
        except Exception:
            return [0] * n, meta

        # Cosine distance matrix for silhouette scoring.
        # dist = 1 - cosine_sim
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
        Xn = X / norms
        sim = Xn @ Xn.T
        dist = 1.0 - sim

        # Choose k by silhouette score over a small range.
        # Dialogue videos: typically 1-4 speakers in short clips.
        max_k = min(6, n)
        best_k = 1
        best_score = -1e9

        for k in range(2, max_k + 1):
            try:
                # sklearn API changed from affinity->metric; handle both.
                try:
                    model = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
                except TypeError:
                    model = AgglomerativeClustering(n_clusters=k, affinity="cosine", linkage="average")
                labs = model.fit_predict(X)
                score = silhouette_score(dist, labs, metric="precomputed")
                meta["candidates"].append({"k": k, "silhouette": float(score)})
                if score > best_score:
                    best_score = score
                    best_k = k
            except Exception:
                logger.warning("Speaker clustering candidate k=%s failed.", k)
                continue

        if best_k == 1:
            # If silhouette failed for all candidates, try a direct k=2 fallback.
            try:
                try:
                    model = AgglomerativeClustering(n_clusters=2, metric="cosine", linkage="average")
                except TypeError:
                    model = AgglomerativeClustering(n_clusters=2, affinity="cosine", linkage="average")
                labs = model.fit_predict(X).tolist()
                meta["chosen_k"] = 2
                meta["best_score"] = float(best_score)
                meta["candidates"].append({"k": 2, "silhouette": float("nan")})
                return labs, meta
            except Exception:
                logger.warning("Speaker clustering fallback k=2 failed.")
            meta["chosen_k"] = 1
            meta["best_score"] = float(best_score)
            return [0] * n, meta

        try:
            try:
                model = AgglomerativeClustering(n_clusters=best_k, metric="cosine", linkage="average")
            except TypeError:
                model = AgglomerativeClustering(n_clusters=best_k, affinity="cosine", linkage="average")
            meta["chosen_k"] = int(best_k)
            meta["best_score"] = float(best_score)
            return model.fit_predict(X).tolist(), meta
        except Exception:
            meta["chosen_k"] = 1
            meta["best_score"] = float(best_score)
            return [0] * n, meta

    def _temporal_smooth_labels(self, labels):
        # Fix isolated flips: A B A -> A A A
        if not labels or len(labels) < 3:
            return labels
        out = list(labels)
        for i in range(1, len(out) - 1):
            if out[i - 1] == out[i + 1] and out[i] != out[i - 1]:
                out[i] = out[i - 1]
        return out

    def _reassign_tiny_clusters(self, labels, embeddings, segments, min_total_seconds=1.2):
        # Clusters with very small total duration are likely noise. Reassign them to the closest big cluster.
        if not labels:
            return labels

        totals = {}
        for lab, seg in zip(labels, segments):
            totals[lab] = totals.get(lab, 0.0) + max(0.01, float(seg.end) - float(seg.start))

        big = {lab for lab, t in totals.items() if t >= float(min_total_seconds)}
        if not big:
            return labels

        # Build centroids for big clusters.
        import numpy as np

        centroids = {}
        counts = {}
        for lab, emb in zip(labels, embeddings):
            if lab not in big:
                continue
            if lab not in centroids:
                centroids[lab] = np.asarray(emb, dtype="float32")
                counts[lab] = 1.0
            else:
                c = counts[lab] + 1.0
                centroids[lab] = (centroids[lab] * (counts[lab] / c)) + (np.asarray(emb, dtype="float32") * (1.0 / c))
                counts[lab] = c

        out = list(labels)
        for i, (lab, emb) in enumerate(zip(labels, embeddings)):
            if lab in big:
                continue
            best_lab = None
            best_sim = -1.0
            for cand, centroid in centroids.items():
                s = cosine_similarity(emb, centroid)
                if s > best_sim:
                    best_sim = s
                    best_lab = cand
            if best_lab is not None:
                out[i] = best_lab

        return out

    def _expand_window(self, start, end, min_seconds, total_duration):
        start = float(start)
        end = float(end)
        total_duration = float(total_duration)
        min_seconds = float(min_seconds)
        if end <= start:
            return max(0.0, start), min(total_duration, start + max(0.1, min_seconds))

        dur = end - start
        if dur >= min_seconds:
            return max(0.0, start), min(total_duration, end)

        extra = (min_seconds - dur) / 2.0
        s = max(0.0, start - extra)
        e = min(total_duration, end + extra)
        # If we hit boundaries, extend on the other side if possible.
        if (e - s) < min_seconds:
            missing = min_seconds - (e - s)
            s = max(0.0, s - missing)
            e = min(total_duration, e + missing)
        if e <= s:
            e = min(total_duration, s + max(0.1, min_seconds))
        return s, e
