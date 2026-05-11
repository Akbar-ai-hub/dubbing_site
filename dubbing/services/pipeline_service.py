import logging
import re
import textwrap
from pathlib import Path

import numpy as np

from .demucs_service import DemucsService
from .ffmpeg_service import FFmpegService
from .prosody_service import ProsodyDurationMapperService
from .speaker_service import SpeakerEmbeddingService, cosine_similarity
from .translation_service import LocalNLLBTranslationService
from .tts_service import CoquiTTSService
from .whisper_service import WhisperService


logger = logging.getLogger(__name__)


class DubbingPipelineService:
    """
    Current pipeline behavior:
    1) Extract audio from video.
    2) Run Demucs once to split vocals/background.
    3) Send the clean vocals track to Whisper and collect timestamped segments.
    4) Translate each Whisper segment independently.
    5) Build an SRT file from the segment timestamps and translated text.
    6) Extract per-segment reference audio from the clean vocals track.
    7) Run XTTS for each segment using the translated text + that segment audio as reference.
    8) Fit each synthesized segment into its original time window and mix on the timeline.
    9) Add the separated background back at -20 dB.
    10) Mux the final audio back into the source video.
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
        diarization_model_name=None,
        hf_auth_token=None,
        diarization_min_speakers=None,
        diarization_max_speakers=None,
        tts_xtts_local_dir=None,
        tts_target_language="",
        xtts_temperature=0.5,
        xtts_length_penalty=1.0,
        xtts_repetition_penalty=3.0,
        xtts_top_k=50,
        xtts_top_p=0.9,
        min_segment_duration=0.4,
        max_speaker_gap_seconds=0.2,
        min_tempo_factor=0.90,
        max_tempo_factor=2.5,
        speaker_ref_target_seconds=8.0,
        speaker_ref_min_seconds=6.0,
        speaker_ref_max_seconds=12.0,
        speaker_ref_min_segment_seconds=0.75,
        speaker_embedding_model_name="pyannote/embedding",
        speaker_embedding_cache_dir=None,
        speaker_embedding_min_seconds=2.0,
        speaker_embedding_device=None,
        min_segment_chars=35,
        max_merge_chars=220,
        ref_audio_output_dir=None,
        subtitle_basename=None,
    ):
        self.ffmpeg = FFmpegService(ffmpeg_bin=ffmpeg_bin)
        self.demucs = DemucsService()
        self.whisper = WhisperService(model_name=whisper_model_name, api_key=groq_api_key)
        self.speaker_embedding = SpeakerEmbeddingService(
            model_name=speaker_embedding_model_name,
            cache_dir=speaker_embedding_cache_dir,
            auth_token=hf_auth_token,
            device=speaker_embedding_device,
            local_files_only=True,
        )
        self.translation = LocalNLLBTranslationService(
            model_dir=translation_model_dir,
            source_lang=nllb_source_lang,
            target_lang=nllb_target_lang,
            batch_size=nllb_batch_size,
            max_new_tokens=nllb_max_new_tokens,
        )
        self.tts = CoquiTTSService(
            xtts_local_dir=tts_xtts_local_dir,
            temperature=xtts_temperature,
            length_penalty=xtts_length_penalty,
            repetition_penalty=xtts_repetition_penalty,
            top_k=xtts_top_k,
            top_p=xtts_top_p,
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
        self.min_segment_duration = max(0.0, float(min_segment_duration))
        self.speaker_embedding_min_seconds = max(0.1, float(speaker_embedding_min_seconds))
        self.min_chars = max(1, int(min_segment_chars))
        self.max_chars = max(40, int(max_merge_chars))
        self.ref_audio_output_dir = (
            str(ref_audio_output_dir).strip() if ref_audio_output_dir else ""
        ) or None
        self.subtitle_basename = (str(subtitle_basename).strip() if subtitle_basename else "") or None

    def run(
        self,
        input_video_path,
        extracted_audio_path,
        tts_audio_path,
        output_video_path,
        progress_callback=None,
    ):
        work_dir = Path(extracted_audio_path).resolve().parent

        self._emit_progress(progress_callback, 5, "extract_audio")
        self.ffmpeg.extract_audio(input_video_path=input_video_path, output_audio_path=extracted_audio_path)
        total_duration = self.ffmpeg.get_duration(extracted_audio_path)

        self._emit_progress(progress_callback, 15, "demucs")
        demucs_out = work_dir / "demucs_main"
        vocals_path, background_path = self.demucs.separate_vocals_and_background(
            extracted_audio_path,
            demucs_out,
        )

        self._emit_progress(progress_callback, 28, "transcription")
        asr = self.whisper.transcribe(vocals_path, language=self.source_language)
        detected_language = asr.get("language")
        words = self._normalize_word_timestamps(asr.get("words") or [], total_duration)
        if not words:
            raise RuntimeError("Whisper returned no usable word timestamps")
        whisper_raw_segments = self._normalize_whisper_segments(asr.get("segments") or [], total_duration)
        whisper_segments = self._regroup_words_into_segments(
            words,
            total_duration,
            vocals_path=vocals_path,
            work_dir=work_dir,
            whisper_raw_segments=whisper_raw_segments,
        )
        if not whisper_segments:
            raise RuntimeError("Whisper returned no usable segments")

        self._emit_progress(progress_callback, 40, "translation")
        translations = []
        transcript_text = []
        translated_text = []
        for segment in whisper_segments:
            src_text = (segment.get("text") or "").strip()
            translation = self.translation.translate(src_text)
            translations.append(translation)
            if src_text:
                transcript_text.append(src_text)
            if translation:
                translated_text.append(translation)

        self._write_sidecar_srt(whisper_segments, translations)

        self._emit_progress(progress_callback, 52, "segment_audio")
        segment_items = self._extract_segment_audio(
            segments=whisper_segments,
            vocals_path=vocals_path,
            work_dir=work_dir,
        )

        self._emit_progress(progress_callback, 60, "tts")
        timeline_segments = []
        total_segments = max(1, len(segment_items))
        for idx, item in enumerate(segment_items):
            translated = translations[idx]
            seg_tts_path = str(work_dir / f"tts_{idx:04d}.wav")
            seg_mapped_path = str(work_dir / f"tts_{idx:04d}_mapped.wav")

            self.tts.synthesize_to_file(
                text=translated,
                output_audio_path=seg_tts_path,
                speaker_reference_audio=item["reference_path"],
                language=self.tts_target_language,
            )

            self.prosody.map_to_duration(
                input_audio_path=seg_tts_path,
                output_audio_path=seg_mapped_path,
                target_duration_sec=item["duration_sec"],
            )
            timeline_segments.append({"path": seg_mapped_path, "start": item["start"]})

            tts_progress = 60 + int(((idx + 1) / total_segments) * 25)
            self._emit_progress(progress_callback, tts_progress, "tts")

        self._emit_progress(progress_callback, 88, "audio_mix")
        self.ffmpeg.mix_segments_on_timeline(
            segments=timeline_segments,
            output_audio_path=tts_audio_path,
            total_duration_sec=total_duration,
        )

        self._emit_progress(progress_callback, 93, "background_mix")
        mixed_tts_path = str(Path(tts_audio_path).with_name("tts_with_bg.wav"))
        self.ffmpeg.mix_with_background(
            foreground_audio_path=tts_audio_path,
            background_audio_path=background_path,
            output_audio_path=mixed_tts_path,
            bg_gain_db=-20,
        )

        self._emit_progress(progress_callback, 97, "video_mux")
        self.ffmpeg.mux_audio_with_video(
            input_video_path=input_video_path,
            input_audio_path=mixed_tts_path,
            output_video_path=output_video_path,
        )

        return {
            "detected_language": detected_language,
            "segments_count": len(whisper_segments),
            "transcript_text": " ".join(transcript_text).strip(),
            "translated_text": " ".join(translated_text).strip(),
            "speaker_references": {
                f"SEGMENT_{idx:04d}": item["reference_path"] for idx, item in enumerate(segment_items)
            },
        }

    def _emit_progress(self, callback, percent, stage):
        if not callback:
            return
        try:
            callback(max(0, min(99, int(percent))), stage)
        except Exception as exc:
            logger.warning("Progress callback failed at stage %s: %s", stage, exc)

    def _normalize_word_timestamps(self, raw_words, total_duration):
        normalized = []
        total_duration = max(0.1, float(total_duration))

        for word in raw_words:
            if not isinstance(word, dict):
                continue
            text = (word.get("word") or word.get("text") or "").strip()
            if not text:
                continue
            try:
                start = max(0.0, float(word.get("start", 0.0)))
                end = min(total_duration, float(word.get("end", 0.0)))
            except (TypeError, ValueError):
                continue

            if end <= start:
                continue
            normalized.append({"word": text, "start": start, "end": end})

        return normalized

    def _normalize_whisper_segments(self, raw_segments, total_duration):
        normalized = []
        total_duration = max(0.1, float(total_duration))
        for idx, segment in enumerate(raw_segments):
            if not isinstance(segment, dict):
                continue
            text = (segment.get("text") or "").strip()
            if not text:
                continue
            try:
                start = max(0.0, float(segment.get("start", 0.0)))
                end = min(total_duration, float(segment.get("end", 0.0)))
            except (TypeError, ValueError):
                continue
            if end <= start:
                continue
            normalized.append({"id": idx, "start": start, "end": end, "text": text})
        return normalized

    def _regroup_words_into_segments(self, words, total_duration, vocals_path, work_dir, whisper_raw_segments=None):
        whisper_raw_segments = whisper_raw_segments or []
        units = []
        current = None
        sentence_end_re = re.compile(r'[.!?]["\')\]]*$')

        for word in words:
            item = dict(word)
            item["source_segment_id"] = self._resolve_source_segment_id(word, whisper_raw_segments)
            if current is None:
                current = self._start_unit(item)
                continue
            tentative_text = self._join_token(current["text"], item["word"])
            gap = max(0.0, float(item["start"]) - float(current["end"]))
            should_split = False
            if sentence_end_re.search(current["text"]):
                should_split = True
            elif len(tentative_text) > self.max_chars and len(current["text"]) >= self.min_chars:
                should_split = True
            elif gap >= 0.85 and len(current["text"]) >= self.min_chars:
                should_split = True

            if should_split:
                self._finalize_unit(units, current)
                current = self._start_unit(item)
                continue

            current["text"] = tentative_text
            current["end"] = float(item["end"])
            current["words"].append(item)

        self._finalize_unit(units, current)
        speaker_labeled_units = self._assign_unit_speakers_with_embeddings(units, vocals_path, work_dir)
        merged = self._merge_short_units(speaker_labeled_units)
        logger.info("Regrouped %d words into %d segments", len(words), len(merged))
        for idx, segment in enumerate(merged):
            logger.info(
                "Segment %d speaker=%s start=%.3f end=%.3f text=%s",
                idx,
                segment.get("speaker") or "SPEAKER_00",
                float(segment.get("start", 0.0)),
                float(segment.get("end", 0.0)),
                segment.get("text") or "",
            )
        return self._normalize_regrouped_segments(merged, total_duration)

    def _assign_unit_speakers_with_embeddings(self, units, vocals_path, work_dir):
        if not units:
            return []

        emb_items = []
        for idx, unit in enumerate(units):
            duration = max(0.0, float(unit["end"]) - float(unit["start"]))
            embed_path = str(Path(work_dir) / f"embed_unit_{idx:04d}.wav")
            self.ffmpeg.extract_audio_segment(
                input_audio_path=vocals_path,
                output_audio_path=embed_path,
                start_sec=float(unit["start"]),
                end_sec=float(unit["end"]),
            )
            try:
                emb = self.speaker_embedding.embed(embed_path)
            finally:
                Path(embed_path).unlink(missing_ok=True)
            emb_items.append(
                {
                    "index": idx,
                    "embedding": emb,
                    "start": float(unit["start"]),
                    "duration_sec": duration,
                }
            )

        target_clusters = self._estimate_speaker_cluster_count(emb_items)
        assignments = self._cluster_speaker_embeddings(emb_items, target_clusters)
        cluster_order = []
        for unit_idx, cluster_idx in sorted(assignments.items(), key=lambda pair: units[pair[0]]["start"]):
            if cluster_idx not in cluster_order:
                cluster_order.append(cluster_idx)
        cluster_to_name = {cluster_idx: f"SPEAKER_{order:02d}" for order, cluster_idx in enumerate(cluster_order)}
        logger.info("Embedding-based sentence speaker assignment: %s", cluster_to_name)

        assigned_units = []
        for idx, unit in enumerate(units):
            cluster_idx = assignments.get(idx, 0)
            assigned = dict(unit)
            assigned["speaker"] = cluster_to_name.get(cluster_idx, f"SPEAKER_{cluster_idx:02d}")
            logger.info(
                "Sentence unit start=%.3f end=%.3f speaker=%s cluster=%s text=%s",
                float(assigned.get("start", 0.0)),
                float(assigned.get("end", 0.0)),
                assigned["speaker"],
                cluster_idx,
                assigned.get("text") or "",
            )
            assigned_units.append(assigned)
        return assigned_units

    def _estimate_speaker_cluster_count(self, emb_items):
        if len(emb_items) <= 1:
            return 1
        vectors = [self._normalize_embedding(item["embedding"]) for item in emb_items]
        similarities = []
        for idx in range(len(vectors)):
            for jdx in range(idx + 1, len(vectors)):
                similarities.append(cosine_similarity(vectors[idx], vectors[jdx]))
        if not similarities:
            return 1
        min_similarity = min(similarities)
        return 2 if min_similarity < 0.80 else 1

    def _resolve_source_segment_id(self, word, whisper_raw_segments):
        if not whisper_raw_segments:
            return None
        best_segment_id = None
        best_overlap = 0.0
        word_start = float(word["start"])
        word_end = float(word["end"])
        word_mid = (word_start + word_end) / 2.0
        for segment in whisper_raw_segments:
            seg_start = float(segment["start"])
            seg_end = float(segment["end"])
            overlap = max(0.0, min(word_end, seg_end) - max(word_start, seg_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_segment_id = segment["id"]
        if best_segment_id is not None:
            return best_segment_id
        for segment in whisper_raw_segments:
            if float(segment["start"]) <= word_mid <= float(segment["end"]):
                return segment["id"]
        return None

    def _start_unit(self, word):
        return {
            "start": float(word["start"]),
            "end": float(word["end"]),
            "text": word["word"],
            "words": [word],
        }

    def _finalize_unit(self, units, unit):
        if not unit:
            return
        text = " ".join((unit.get("text") or "").split()).strip()
        if not text:
            return
        units.append(
            {
                "start": float(unit["start"]),
                "end": float(unit["end"]),
                "text": text,
                "words": list(unit["words"]),
                "source_segment_ids": sorted(
                    {
                        item.get("source_segment_id")
                        for item in (unit.get("words") or [])
                        if item.get("source_segment_id") is not None
                    }
                ),
            }
        )

    def _merge_short_units(self, units):
        merged = [dict(unit) for unit in units]
        idx = 0
        while idx < len(merged):
            unit = merged[idx]
            if len(unit["text"]) >= self.min_chars:
                idx += 1
                continue

            next_idx = idx + 1
            prev_idx = idx - 1
            can_merge_next = (
                next_idx < len(merged)
                and merged[next_idx]["speaker"] == unit["speaker"]
                and len(unit["text"]) + 1 + len(merged[next_idx]["text"]) <= self.max_chars
            )
            can_merge_prev = (
                prev_idx >= 0
                and merged[prev_idx]["speaker"] == unit["speaker"]
                and len(merged[prev_idx]["text"]) + 1 + len(unit["text"]) <= self.max_chars
            )

            if can_merge_next:
                merged[next_idx] = self._combine_units(unit, merged[next_idx])
                del merged[idx]
                continue
            if can_merge_prev:
                merged[prev_idx] = self._combine_units(merged[prev_idx], unit)
                del merged[idx]
                idx = max(0, idx - 1)
                continue
            idx += 1
        return merged

    def _combine_units(self, left, right):
        return {
            "speaker": left["speaker"],
            "start": float(left["start"]),
            "end": float(right["end"]),
            "text": self._join_token(left["text"], right["text"]),
            "words": list(left["words"]) + list(right["words"]),
        }

    def _normalize_regrouped_segments(self, segments, total_duration):
        total_duration = max(0.1, float(total_duration))
        normalized = []
        for segment in segments:
            start = max(0.0, float(segment.get("start", 0.0)))
            end = min(total_duration, float(segment.get("end", 0.0)))
            text = (segment.get("text") or "").strip()
            if not text or end <= start:
                continue
            if (end - start) <= self.min_segment_duration:
                continue
            normalized.append(
                {
                    "start": start,
                    "end": end,
                    "text": text,
                    "speaker": segment.get("speaker") or "SPEAKER_00",
                    "char_count": len(text),
                    "words": list(segment.get("words") or []),
                    "source_segment_ids": list(segment.get("source_segment_ids") or []),
                }
            )
        return normalized

    def _join_token(self, left, right):
        left = (left or "").strip()
        right = (right or "").strip()
        if not left:
            return right
        if not right:
            return left
        no_space_prefix = {".", ",", "!", "?", ";", ":", "%", ")", "]", "}", "'s"}
        no_space_suffix = {"(", "[", "{", '"', "'"}
        if right in no_space_prefix or right.startswith("'"):
            return f"{left}{right}"
        if left.endswith(tuple(no_space_suffix)):
            return f"{left}{right}"
        return f"{left} {right}"

    def _extract_segment_audio(self, segments, vocals_path, work_dir):
        items = []
        for idx, segment in enumerate(segments):
            start = float(segment["start"])
            end = float(segment["end"])
            duration = max(0.1, end - start)

            raw_ref_path = str(Path(work_dir) / f"segment_{idx:04d}_raw.wav")
            denoised_ref_path = str(Path(work_dir) / f"segment_{idx:04d}_denoised.wav")
            ref_22050_path = str(Path(work_dir) / f"segment_{idx:04d}_ref_22050.wav")

            self.ffmpeg.extract_audio_segment(
                input_audio_path=vocals_path,
                output_audio_path=raw_ref_path,
                start_sec=start,
                end_sec=end,
            )
            self.ffmpeg.denoise_audio(
                input_audio_path=raw_ref_path,
                output_audio_path=denoised_ref_path,
            )
            self.ffmpeg.resample_audio(
                input_audio_path=denoised_ref_path,
                output_audio_path=ref_22050_path,
                sample_rate_hz=22050,
            )

            items.append(
                {
                    "start": start,
                    "end": end,
                    "duration_sec": duration,
                    "speaker": segment.get("speaker") or "SPEAKER_00",
                    "text": segment.get("text") or "",
                    "char_count": int(segment.get("char_count") or 0),
                    "reference_path": ref_22050_path,
                    "source_reference_path": ref_22050_path,
                }
            )

        self._assign_reference_fallbacks(items)
        return items

    def _cluster_speaker_embeddings(self, emb_items, target_clusters):
        if not emb_items:
            return {}

        vectors = [self._normalize_embedding(item["embedding"]) for item in emb_items]
        k = max(1, min(int(target_clusters), len(vectors)))
        seed_indexes = [max(range(len(emb_items)), key=lambda idx: emb_items[idx]["duration_sec"])]
        while len(seed_indexes) < k:
            best_idx = None
            best_distance = None
            for idx, vector in enumerate(vectors):
                if idx in seed_indexes:
                    continue
                min_similarity = max(cosine_similarity(vector, vectors[seed_idx]) for seed_idx in seed_indexes)
                distance = 1.0 - min_similarity
                if best_distance is None or distance > best_distance:
                    best_distance = distance
                    best_idx = idx
            if best_idx is None:
                break
            seed_indexes.append(best_idx)

        centroids = [vectors[idx].copy() for idx in seed_indexes]
        assignments = {}
        for _ in range(12):
            updated_assignments = {}
            for idx, vector in enumerate(vectors):
                scores = [cosine_similarity(vector, centroid) for centroid in centroids]
                updated_assignments[idx] = int(max(range(len(scores)), key=lambda cluster_idx: scores[cluster_idx]))
            if updated_assignments == assignments:
                break
            assignments = updated_assignments

            new_centroids = []
            for cluster_idx in range(len(centroids)):
                members = [vectors[idx] for idx, assigned in assignments.items() if assigned == cluster_idx]
                if not members:
                    new_centroids.append(centroids[cluster_idx])
                    continue
                mean_vec = np.mean(np.stack(members, axis=0), axis=0)
                new_centroids.append(self._normalize_embedding(mean_vec))
            centroids = new_centroids

        return {emb_items[idx]["index"]: cluster_idx for idx, cluster_idx in assignments.items()}

    def _normalize_embedding(self, vector):
        arr = np.asarray(vector, dtype="float32").reshape(-1)
        norm = np.linalg.norm(arr) + 1e-8
        return arr / norm

    def _assign_reference_fallbacks(self, segment_items):
        if not segment_items:
            return

        strong_candidates = [item for item in segment_items if item["char_count"] >= self.min_chars]
        for idx, item in enumerate(segment_items):
            if item["char_count"] >= self.min_chars:
                continue
            fallback = self._find_reference_fallback(idx, segment_items, strong_candidates)
            if fallback:
                item["reference_path"] = fallback["source_reference_path"]

    def _find_reference_fallback(self, idx, segment_items, strong_candidates):
        item = segment_items[idx]
        same_speaker = [
            candidate
            for candidate in strong_candidates
            if candidate["speaker"] == item["speaker"] and candidate["source_reference_path"] != item["source_reference_path"]
        ]
        pool = same_speaker or [
            candidate for candidate in strong_candidates if candidate["source_reference_path"] != item["source_reference_path"]
        ]
        if not pool:
            return None

        target_mid = (float(item["start"]) + float(item["end"])) / 2.0
        return min(
            pool,
            key=lambda candidate: abs(((float(candidate["start"]) + float(candidate["end"])) / 2.0) - target_mid),
        )

    def _write_sidecar_srt(self, segments, translations):
        if not self.ref_audio_output_dir or not self.subtitle_basename:
            return
        if not segments or not translations:
            return

        out_dir = Path(self.ref_audio_output_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / self.subtitle_basename

        lines = []
        for idx, (segment, text) in enumerate(zip(segments, translations), start=1):
            start = float(segment.get("start", 0.0))
            end = float(segment.get("end", 0.0))
            if end <= start:
                end = start + 0.2
            lines.append(str(idx))
            lines.append(f"{self._fmt_srt_ts(start)} --> {self._fmt_srt_ts(end)}")
            lines.extend(self._format_srt_text(text))
            lines.append("")

        with out_path.open("w", encoding="utf-8-sig", newline="") as handle:
            handle.write("\r\n".join(lines) + "\r\n")
        logger.info("SRT subtitle written: %s", out_path)

    def _fmt_srt_ts(self, seconds):
        total_ms = int(max(0.0, float(seconds)) * 1000)
        ms = total_ms % 1000
        total_s = total_ms // 1000
        s = total_s % 60
        total_m = total_s // 60
        m = total_m % 60
        h = total_m // 60
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    def _format_srt_text(self, text, line_width=42):
        normalized = " ".join((text or "").replace("\r", " ").replace("\n", " ").split()).strip()
        if not normalized:
            return [""]
        return textwrap.wrap(
            normalized,
            width=max(10, int(line_width)),
            break_long_words=False,
            break_on_hyphens=False,
        ) or [normalized]
