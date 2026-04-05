import logging
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path


from .ffmpeg_service import FFmpegService
from .demucs_service import DemucsService
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
        tts_target_language="",
        xtts_temperature=0.5,
        xtts_length_penalty=1.0,
        xtts_repetition_penalty=3.0,
        xtts_top_k=50,
        xtts_top_p=0.9,
        # The following knobs default to sane values and are intended to be auto-tuned.
        # We keep them as parameters for code-level control, but do not require .env tuning.
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
        self.max_merge_chars = int(max_merge_chars)
        self.speaker_embedder = SpeakerEmbeddingService(
            model_name=speaker_embedding_model_name,
            cache_dir=speaker_embedding_cache_dir,
            auth_token=hf_auth_token,
            device=speaker_embedding_device,
        )
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
        self._emit_progress(progress_callback, 5, "extract_audio")
        # 1) Extract full audio
        self.ffmpeg.extract_audio(input_video_path=input_video_path, output_audio_path=extracted_audio_path)
        total_duration = self.ffmpeg.get_duration(extracted_audio_path)

        self._emit_progress(progress_callback, 12, "transcription")
        # 2) ASR
        asr = self.whisper.transcribe(extracted_audio_path, language=self.source_language)
        detected_language = asr.get("language")
        whisper_segments = asr.get("segments") or []

        if not whisper_segments:
            raise RuntimeError("Whisper returned no segments")

        self._emit_progress(progress_callback, 22, "segmenting")
        # 3) Split into sentence-like subsegments
        raw_segments = self._split_whisper_segments_into_subsegments(whisper_segments)
        raw_segments = self._merge_too_short_segments(raw_segments, min_dur=self.min_segment_duration)

        self._emit_progress(progress_callback, 32, "speaker_embedding")
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

        self._emit_progress(progress_callback, 45, "merge_segments")
        # 5) Merge adjacent segments by speaker and gap
        merged = self._merge_adjacent_by_speaker(raw_segments, speaker_ids, max_gap=self.max_speaker_gap_seconds)

        self._emit_progress(progress_callback, 55, "translation")
        # 6) Translate in one local batch
        src_texts = [m["text"] for m in merged]
        translations = [self.translation.translate(text) for text in src_texts]
        self._write_sidecar_srt(merged, translations)

        self._emit_progress(progress_callback, 62, "speaker_reference")
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
        total_segments = max(1, len(merged))

        def _window_end(chunk_start_idx, chunk_end_idx):
            start_val = float(merged[chunk_start_idx]["start"])
            end_val = float(merged[chunk_end_idx]["end"])
            if chunk_end_idx + 1 < len(merged):
                next_start_val = float(merged[chunk_end_idx + 1]["start"])
                return next_start_val if next_start_val > start_val else end_val
            return end_val

        synthesized_chunks = []
        idx = 0
        while idx < len(merged):
            m = merged[idx]
            spk = m["speaker"]
            chunk_start_idx = idx
            chunk_end_idx = idx
            start = float(m["start"])
            end = float(m["end"])
            tr_text = translations[idx]
            src_text = m["text"]

            # Each segment must occupy the window [start, next_start) without cutting audio.
            window_end = _window_end(chunk_start_idx, chunk_end_idx)
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

            seg_tts_path = str(work_dir / f"tts_{chunk_start_idx:04d}_{chunk_end_idx:04d}.wav")
            seg_mapped_path = str(work_dir / f"tts_{chunk_start_idx:04d}_{chunk_end_idx:04d}_mapped.wav")

            self.tts.synthesize_to_file(
                text=tr_text,
                output_audio_path=seg_tts_path,
                speaker_reference_audio=ref_path,
                language=self.tts_target_language,
            )
            source_duration = self.ffmpeg.get_duration(seg_tts_path)
            max_speedup = float(self.prosody.max_tempo_factor)
            needed_factor = source_duration / max(0.01, base_window)

            # If too fast, first try merging with previous synthesized chunk of the same speaker.
            merged_with_previous = False
            if needed_factor > max_speedup and synthesized_chunks:
                prev_chunk = synthesized_chunks[-1]
                if (
                    prev_chunk["speaker"] == spk
                    and prev_chunk["end_idx"] == (idx - 1)
                ):
                    candidate_tr_text = (prev_chunk["tr_text"] + " " + tr_text).strip()
                    if len(candidate_tr_text) <= max(1, self.max_merge_chars):
                        merged_with_previous = True
                        synthesized_chunks.pop()
                        if timeline_segments:
                            timeline_segments.pop()
                        if transcript_text:
                            transcript_text.pop()
                        if translated_text:
                            translated_text.pop()
                        chunk_start_idx = prev_chunk["start_idx"]
                        start = float(prev_chunk["start"])
                        src_text = (prev_chunk["src_text"] + " " + src_text).strip()
                        tr_text = candidate_tr_text
                        window_end = _window_end(chunk_start_idx, chunk_end_idx)
                        base_window = max(0.01, window_end - start)
                        seg_tts_path = str(work_dir / f"tts_{chunk_start_idx:04d}_{chunk_end_idx:04d}.wav")
                        seg_mapped_path = str(
                            work_dir / f"tts_{chunk_start_idx:04d}_{chunk_end_idx:04d}_mapped.wav"
                        )
                        self.tts.synthesize_to_file(
                            text=tr_text,
                            output_audio_path=seg_tts_path,
                            speaker_reference_audio=ref_path,
                            language=self.tts_target_language,
                        )
                        source_duration = self.ffmpeg.get_duration(seg_tts_path)
                        needed_factor = source_duration / max(0.01, base_window)

            # If still too fast, try merging with next same-speaker segments while text limit allows.
            while (
                needed_factor > max_speedup
                and (chunk_end_idx + 1) < len(merged)
                and merged[chunk_end_idx + 1]["speaker"] == spk
            ):
                next_idx = chunk_end_idx + 1
                candidate_tr_text = (tr_text + " " + translations[next_idx]).strip()
                if len(candidate_tr_text) > max(1, self.max_merge_chars):
                    break

                chunk_end_idx = next_idx
                src_text = (src_text + " " + merged[chunk_end_idx]["text"]).strip()
                tr_text = candidate_tr_text
                end = float(merged[chunk_end_idx]["end"])
                window_end = _window_end(chunk_start_idx, chunk_end_idx)
                base_window = max(0.01, window_end - start)

                self.tts.synthesize_to_file(
                    text=tr_text,
                    output_audio_path=seg_tts_path,
                    speaker_reference_audio=ref_path,
                    language=self.tts_target_language,
                )
                source_duration = self.ffmpeg.get_duration(seg_tts_path)
                needed_factor = source_duration / max(0.01, base_window)

            if merged_with_previous:
                logger.warning(
                    "Merged segments %d..%d with previous speaker chunk for %s (needed=%.2f max=%.2f, chars=%d).",
                    chunk_start_idx,
                    chunk_end_idx,
                    spk,
                    needed_factor,
                    max_speedup,
                    len(tr_text),
                )
            if chunk_end_idx > idx:
                logger.warning(
                    "Merged segments %d..%d for speaker %s to avoid over-speed (needed=%.2f max=%.2f).",
                    idx,
                    chunk_end_idx,
                    spk,
                    needed_factor,
                    max_speedup,
                )

            self.prosody.map_to_duration(
                input_audio_path=seg_tts_path,
                output_audio_path=seg_mapped_path,
                target_duration_sec=base_window,
            )

            transcript_text.append(src_text)
            translated_text.append(tr_text)

            timeline_segments.append({"path": seg_mapped_path, "start": start})
            synthesized_chunks.append(
                {
                    "start_idx": chunk_start_idx,
                    "end_idx": chunk_end_idx,
                    "speaker": spk,
                    "start": start,
                    "src_text": src_text,
                    "tr_text": tr_text,
                }
            )
            processed_segments = chunk_end_idx + 1
            tts_progress = 62 + int((processed_segments / total_segments) * 28)
            self._emit_progress(progress_callback, tts_progress, "tts")
            idx = chunk_end_idx + 1

        self._emit_progress(progress_callback, 92, "audio_mix")
        self.ffmpeg.mix_segments_on_timeline(
            segments=timeline_segments,
            output_audio_path=tts_audio_path,
            total_duration_sec=total_duration,
        )
        # Mix background (non-vocals) at -20 dB under the dubbed audio.
        mixed_tts_path = str(Path(tts_audio_path).with_name("tts_with_bg.wav"))
        try:
            demucs_out = Path(extracted_audio_path).resolve().parent / "demucs_bg"
            background_path = self.demucs.separate_background(extracted_audio_path, demucs_out)
            self.ffmpeg.mix_with_background(
                foreground_audio_path=tts_audio_path,
                background_audio_path=background_path,
                output_audio_path=mixed_tts_path,
                bg_gain_db=-20,
            )
            final_audio_path = mixed_tts_path
        except Exception as exc:
            logger.warning("Failed to mix background noise, using clean TTS audio: %s", exc)
            final_audio_path = tts_audio_path

        self._emit_progress(progress_callback, 97, "video_mux")
        self.ffmpeg.mux_audio_with_video(
            input_video_path=input_video_path,
            input_audio_path=final_audio_path,
            output_video_path=output_video_path,
        )

        return {
            "detected_language": detected_language,
            "segments_count": len(merged),
            "transcript_text": " ".join([t for t in transcript_text if t]).strip(),
            "translated_text": " ".join([t for t in translated_text if t]).strip(),
            "speaker_references": {k: v["path"] for k, v in speaker_refs.items()},
        }

    def _emit_progress(self, callback, percent, stage):
        if not callback:
            return
        try:
            callback(max(0, min(99, int(percent))), stage)
        except Exception as exc:
            logger.warning("Progress callback failed at stage %s: %s", stage, exc)

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

    def _write_sidecar_srt(self, merged, translations):
        if not self.ref_audio_output_dir or not self.subtitle_basename:
            return
        if not merged or not translations:
            return

        out_dir = Path(self.ref_audio_output_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / self.subtitle_basename

        lines = []
        for idx, (seg, text) in enumerate(zip(merged, translations), start=1):
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", 0.0))
            if end <= start:
                end = start + 0.2
            lines.append(str(idx))
            lines.append(f"{self._fmt_srt_ts(start)} --> {self._fmt_srt_ts(end)}")
            lines.extend(self._format_srt_text(text))
            lines.append("")

        try:
            # Windows media players are much more reliable with UTF-8 BOM + CRLF for Cyrillic SRT.
            with out_path.open("w", encoding="utf-8-sig", newline="") as handle:
                handle.write("\r\n".join(lines) + "\r\n")
            logger.info("SRT subtitle written: %s", out_path)
        except Exception as exc:
            logger.warning("Failed to write SRT subtitle: %s", exc)

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

            # Optional: use demucs to isolate vocals (reduce background music).
            vocals_path = None
            try:
                demucs_out = Path(work_dir) / "demucs"
                vocals_path = self.demucs.separate_vocals(out_path, demucs_out)
                out_path = vocals_path
            except Exception as exc:
                logger.warning("Demucs vocals separation failed for %s: %s", spk, exc)

            # Denoise and resample to 22050 Hz as XTTS reference requirement.
            denoised_path = str(Path(work_dir) / f"ref_{spk}_denoised.wav")
            try:
                self.ffmpeg.denoise_audio(out_path, denoised_path)
                out_path = denoised_path
            except Exception as exc:
                logger.warning("Failed to denoise speaker ref %s: %s", spk, exc)

            resampled_path = str(Path(work_dir) / f"ref_{spk}_denoised_22050.wav")
            try:
                self.ffmpeg.resample_audio(out_path, resampled_path, sample_rate_hz=22050)
                out_path = resampled_path
            except Exception as exc:
                logger.warning("Failed to resample speaker ref %s: %s", spk, exc)
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
