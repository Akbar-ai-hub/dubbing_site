import logging
import re
import textwrap
from pathlib import Path

import numpy as np
import soundfile as sf

from .demucs_service import DemucsService
from .ffmpeg_service import FFmpegService
from .prosody_service import ProsodyDurationMapperService
from .speaker_service import SpeakerAttributionService, SpeakerEmbeddingService
from .translation_service import LocalNLLBTranslationService
from .tts_service import CoquiTTSService
from .whisper_service import WhisperService


logger = logging.getLogger(__name__)


class DubbingPipelineService:
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
        speaker_embedding_model_dir=None,
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
        self.speaker_embedding = SpeakerEmbeddingService(
            model_name=speaker_embedding_model_name,
            cache_dir=speaker_embedding_cache_dir,
            model_dir=speaker_embedding_model_dir,
            auth_token=hf_auth_token,
            device=speaker_embedding_device,
            local_files_only=True,
        )
        preferred_speakers = None
        try:
            if str(diarization_max_speakers).strip():
                preferred_speakers = int(diarization_max_speakers)
        except Exception:
            preferred_speakers = None
        self.speaker_attribution = SpeakerAttributionService(
            embedding_service=self.speaker_embedding,
            auth_token=hf_auth_token,
            default_num_speakers=preferred_speakers,
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
        self._persist_clean_vocals(vocals_path)

        self._emit_progress(progress_callback, 28, "transcription")
        asr = self.whisper.transcribe(extracted_audio_path, language=self.source_language)
        detected_language = asr.get("language")
        raw_words = asr.get("words") or []
        raw_segments = asr.get("segments") or []
        words = self._normalize_word_timestamps(raw_words, total_duration)
        if not words:
            raise RuntimeError("Whisper returned no usable word timestamps")

        whisper_raw_segments = self._normalize_whisper_segments(raw_segments, total_duration)
        leading_speech_onset_sec = self._estimate_leading_speech_onset_for_asr(
            raw_words=raw_words,
            vocals_path=vocals_path,
        )
        whisper_segments = self._build_sentence_segments(
            words=words,
            total_duration=total_duration,
            vocals_path=vocals_path,
            whisper_raw_segments=whisper_raw_segments,
            leading_speech_onset_sec=leading_speech_onset_sec,
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
                source_prosody=item.get("source_prosody"),
                transcript_text=item.get("text") or translated,
            )
            timeline_segments.append({"path": seg_mapped_path, "start": float(item["start"])})

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

    def _normalize_word_timestamps(self, raw_words, total_duration, offset_sec=0.0):
        normalized = []
        total_duration = max(0.1, float(total_duration))
        offset_sec = float(offset_sec)
        for word in raw_words:
            if not isinstance(word, dict):
                continue
            text = (word.get("word") or word.get("text") or "").strip()
            if not text:
                continue
            try:
                start = max(0.0, float(word.get("start", 0.0)) + offset_sec)
                end = min(total_duration, float(word.get("end", 0.0)) + offset_sec)
            except (TypeError, ValueError):
                continue
            if end <= start:
                continue
            normalized.append({"word": text, "start": start, "end": end})
        return normalized

    def _normalize_whisper_segments(self, raw_segments, total_duration, offset_sec=0.0):
        normalized = []
        total_duration = max(0.1, float(total_duration))
        offset_sec = float(offset_sec)
        for idx, segment in enumerate(raw_segments):
            if not isinstance(segment, dict):
                continue
            text = (segment.get("text") or "").strip()
            if not text:
                continue
            try:
                start = max(0.0, float(segment.get("start", 0.0)) + offset_sec)
                end = min(total_duration, float(segment.get("end", 0.0)) + offset_sec)
            except (TypeError, ValueError):
                continue
            if end <= start:
                continue
            normalized.append({"id": idx, "start": start, "end": end, "text": text})
        return normalized

    def _estimate_leading_speech_onset_for_asr(self, raw_words, vocals_path):
        asr_start = self._first_asr_timestamp(raw_words, [])
        if asr_start is None or float(asr_start) > 0.35:
            return 0.0
        try:
            speech_onset = self._estimate_first_speech_onset_sec(vocals_path)
        except Exception as exc:
            logger.warning("Failed to estimate leading speech onset: %s", exc)
            return 0.0
        if speech_onset < 1.0:
            return 0.0
        logger.info("Detected leading speech onset for first ASR segment: %.3f", speech_onset)
        return float(speech_onset)

    def _estimate_asr_timestamp_offset(self, raw_words, raw_segments, vocals_path):
        asr_start = self._first_asr_timestamp(raw_words, raw_segments)
        if asr_start is None:
            return 0.0

        try:
            speech_onset = self._estimate_first_speech_onset_sec(vocals_path)
        except Exception as exc:
            logger.warning("Failed to estimate first speech onset for timestamp alignment: %s", exc)
            return 0.0

        offset = float(speech_onset) - float(asr_start)
        if abs(offset) < 0.50:
            logger.info(
                "ASR timestamp offset skipped: speech_onset=%.3f asr_start=%.3f offset=%.3f",
                speech_onset,
                asr_start,
                offset,
            )
            return 0.0

        logger.info(
            "Applying ASR timestamp offset: speech_onset=%.3f asr_start=%.3f offset=%.3f",
            speech_onset,
            asr_start,
            offset,
        )
        return offset

    def _first_asr_timestamp(self, raw_words, raw_segments):
        starts = []
        for item in raw_words or []:
            if not isinstance(item, dict):
                continue
            try:
                starts.append(float(item.get("start", 0.0)))
            except (TypeError, ValueError):
                continue
        for item in raw_segments or []:
            if not isinstance(item, dict):
                continue
            try:
                starts.append(float(item.get("start", 0.0)))
            except (TypeError, ValueError):
                continue
        starts = [value for value in starts if value >= 0.0]
        return min(starts) if starts else None

    def _estimate_first_speech_onset_sec(self, audio_path):
        samples, sample_rate = sf.read(str(audio_path), always_2d=False)
        if isinstance(samples, np.ndarray) and samples.ndim > 1:
            samples = np.mean(samples, axis=1)
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            return 0.0

        frame_len = max(1, int(sample_rate * 0.05))
        hop = max(1, int(sample_rate * 0.01))
        rms_values = []
        for start in range(0, max(1, len(samples) - frame_len + 1), hop):
            frame = samples[start:start + frame_len]
            if frame.size < frame_len:
                break
            rms_values.append(float(np.sqrt(np.mean(frame * frame) + 1e-9)))
        if not rms_values:
            return 0.0

        rms = np.asarray(rms_values, dtype=np.float32)
        peak = float(np.max(rms))
        if peak <= 0.0:
            return 0.0

        threshold = max(peak * 0.10, float(np.percentile(rms, 70)) * 1.8, 5e-4)
        frame_sec = hop / float(sample_rate)
        consecutive_needed = max(2, int(round(0.10 / max(frame_sec, 1e-6))))
        streak = 0
        for idx, value in enumerate(rms):
            if float(value) >= threshold:
                streak += 1
                if streak >= consecutive_needed:
                    onset_frame = idx - streak + 1
                    return max(0.0, onset_frame * frame_sec)
            else:
                streak = 0
        return 0.0


    def _build_sentence_segments(
        self,
        words,
        total_duration,
        vocals_path,
        whisper_raw_segments=None,
        leading_speech_onset_sec=0.0,
    ):
        whisper_raw_segments = whisper_raw_segments or []
        units = []
        current = None
        sentence_end_re = re.compile(r'[.!?]["\')\]]*$')
        leading_speech_onset_sec = max(0.0, float(leading_speech_onset_sec or 0.0))

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
            elif gap >= 0.60:
                # Large pauses should always start a new segment, even for short utterances.
                should_split = True
            elif len(tentative_text) > self.max_chars and len(current["text"]) >= self.min_chars:
                should_split = True

            if should_split:
                self._finalize_unit(units, current)
                current = self._start_unit(item)
                continue

            current["text"] = tentative_text
            current["end"] = float(item["end"])
            current["words"].append(item)

        self._finalize_unit(units, current)
        speaker_labeled_units = self.speaker_attribution.assign_speakers(
            segments=units,
            audio_path=vocals_path,
        )
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
        if leading_speech_onset_sec > 0.0:
            merged = self._apply_leading_speech_onset_to_first_segment(merged, leading_speech_onset_sec)
        return self._normalize_regrouped_segments(merged, total_duration)

    def _apply_leading_speech_onset_to_first_segment(self, segments, leading_speech_onset_sec):
        if not segments:
            return segments
        first = dict(segments[0])
        start = float(first.get("start", 0.0))
        end = float(first.get("end", 0.0))
        leading_speech_onset_sec = float(leading_speech_onset_sec)
        if start <= 0.35 and start < leading_speech_onset_sec < end:
            logger.info(
                "Adjusting first segment start from %.3f to leading speech onset %.3f",
                start,
                leading_speech_onset_sec,
            )
            first["start"] = leading_speech_onset_sec
            return [first] + [dict(segment) for segment in segments[1:]]
        return segments

    def _persist_clean_vocals(self, vocals_path):
        if not self.ref_audio_output_dir:
            return

        src = Path(vocals_path)
        if not src.exists():
            return

        out_dir = Path(self.ref_audio_output_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        base_name = "clean_vocals.wav"
        if self.subtitle_basename:
            subtitle_name = Path(self.subtitle_basename).name
            if subtitle_name.lower().endswith(".srt"):
                base_name = subtitle_name[:-4] + "_clean_vocals.wav"
        dest = out_dir / base_name
        try:
            dest.write_bytes(src.read_bytes())
            logger.info("Clean vocals audio copied to: %s", dest)
        except Exception as exc:
            logger.warning("Failed to persist clean vocals audio %s: %s", src, exc)

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
            "words": list(left.get("words") or []) + list(right.get("words") or []),
            "source_segment_ids": sorted(
                set(left.get("source_segment_ids") or []) | set(right.get("source_segment_ids") or [])
            ),
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
        ref_source_path = self._prepare_reference_source(vocals_path, work_dir)
        items = []
        for idx, segment in enumerate(segments):
            start = float(segment["start"])
            end = float(segment["end"])
            duration = max(0.1, end - start)

            raw_ref_path = str(Path(work_dir) / f"segment_{idx:04d}_raw.wav")
            ref_22050_path = str(Path(work_dir) / f"segment_{idx:04d}_ref_22050.wav")

            self.ffmpeg.extract_audio_segment(
                input_audio_path=vocals_path,
                output_audio_path=raw_ref_path,
                start_sec=start,
                end_sec=end,
            )
            self.ffmpeg.extract_audio_segment(
                input_audio_path=ref_source_path,
                output_audio_path=ref_22050_path,
                start_sec=start,
                end_sec=end,
                sample_rate_hz=22050,
            )
            source_prosody = self.prosody.analyze_source_segment(
                input_audio_path=raw_ref_path,
                transcript_text=segment.get("text") or "",
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
                    "segment_reference_path": ref_22050_path,
                    "reference_duration_sec": duration,
                    "source_prosody": source_prosody,
                }
            )

        self._assign_reference_audio(items, work_dir)
        return items

    def _prepare_reference_source(self, vocals_path, work_dir):
        denoised_path = str(Path(work_dir) / "reference_source_denoised.wav")
        ref_source_path = str(Path(work_dir) / "reference_source_22050.wav")
        self.ffmpeg.denoise_audio(
            input_audio_path=vocals_path,
            output_audio_path=denoised_path,
            sample_rate_hz=22050,
        )
        self.ffmpeg.resample_audio(
            input_audio_path=denoised_path,
            output_audio_path=ref_source_path,
            sample_rate_hz=22050,
        )
        return ref_source_path

    def _assign_reference_audio(self, segment_items, work_dir):
        if not segment_items:
            return

        self._combine_short_reference_audio(segment_items, work_dir, min_duration_sec=3.0)

    def _combine_short_reference_audio(self, segment_items, work_dir, min_duration_sec=3.0):
        min_duration_sec = float(min_duration_sec)
        for idx, item in enumerate(segment_items):
            own_duration = float(item.get("reference_duration_sec") or 0.0)
            if own_duration >= min_duration_sec:
                continue

            parts = self._collect_reference_parts(idx, segment_items, min_duration_sec)
            total_duration = sum(float(part.get("reference_duration_sec") or 0.0) for part in parts)
            if len(parts) <= 1 or total_duration < min_duration_sec:
                logger.warning(
                    "Reference audio for segment %s is %.3fs and no same-speaker audio can raise it to %.3fs",
                    idx,
                    own_duration,
                    min_duration_sec,
                )
                continue

            output_path = str(Path(work_dir) / f"segment_{idx:04d}_ref_combined_22050.wav")
            ordered_parts = sorted(parts, key=lambda part: float(part["start"]))
            self.ffmpeg.concat_audio_files(
                [part["segment_reference_path"] for part in ordered_parts],
                output_audio_path=output_path,
                sample_rate_hz=22050,
            )
            item["reference_path"] = output_path
            item["source_reference_path"] = output_path
            item["reference_duration_sec"] = total_duration
            logger.info(
                "Combined same-speaker reference for segment %s speaker=%s duration=%.3fs parts=%s",
                idx,
                item.get("speaker"),
                total_duration,
                [segment_items.index(part) for part in ordered_parts],
            )

    def _collect_reference_parts(self, idx, segment_items, min_duration_sec):
        item = segment_items[idx]
        target_mid = (float(item["start"]) + float(item["end"])) / 2.0
        parts = [item]
        total_duration = float(item.get("reference_duration_sec") or 0.0)

        candidates = [
            candidate
            for candidate_idx, candidate in enumerate(segment_items)
            if candidate_idx != idx and candidate.get("speaker") == item.get("speaker")
        ]
        candidates.sort(
            key=lambda candidate: (
                abs(((float(candidate["start"]) + float(candidate["end"])) / 2.0) - target_mid),
                -float(candidate.get("reference_duration_sec") or 0.0),
            )
        )

        for candidate in candidates:
            parts.append(candidate)
            total_duration += float(candidate.get("reference_duration_sec") or 0.0)
            if total_duration >= min_duration_sec:
                break

        return parts

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
