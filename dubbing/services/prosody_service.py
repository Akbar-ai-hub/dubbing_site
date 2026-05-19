from pathlib import Path
import logging
import math

import numpy as np
import soundfile as sf


logger = logging.getLogger(__name__)


class ProsodyDurationMapperService:
    def __init__(self, ffmpeg_service, min_tempo_factor=0.85, max_tempo_factor=1.15):
        self.ffmpeg_service = ffmpeg_service
        self.min_tempo_factor = float(min_tempo_factor)
        self.max_tempo_factor = float(max_tempo_factor)
        if self.min_tempo_factor <= 0 or self.max_tempo_factor <= 0:
            raise RuntimeError("Tempo factor bounds must be positive")
        if self.min_tempo_factor > self.max_tempo_factor:
            raise RuntimeError("min_tempo_factor must be <= max_tempo_factor")

    def analyze_source_segment(self, input_audio_path, transcript_text=""):
        samples, sample_rate = self._load_audio(input_audio_path)
        duration_sec = len(samples) / max(1, sample_rate)
        words = self._count_words(transcript_text)
        energy_db = self._estimate_rms_db(samples)
        pause_profile = self._estimate_pause_profile(samples, sample_rate)
        pitch_profile = self._estimate_pitch_profile(samples, sample_rate)

        voiced_duration = max(0.05, duration_sec - pause_profile["total_silence_sec"])
        speaking_rate_wps = words / voiced_duration if words > 0 else 0.0
        profile = {
            "duration_sec": float(duration_sec),
            "word_count": int(words),
            "energy_db": float(energy_db),
            "speaking_rate_wps": float(speaking_rate_wps),
            "pause_ratio": float(pause_profile["pause_ratio"]),
            "pause_count": int(pause_profile["pause_count"]),
            "leading_silence_sec": float(pause_profile["leading_silence_sec"]),
            "trailing_silence_sec": float(pause_profile["trailing_silence_sec"]),
            "longest_pause_sec": float(pause_profile["longest_pause_sec"]),
            "pitch_mean_hz": float(pitch_profile["pitch_mean_hz"]),
            "pitch_range_hz": float(pitch_profile["pitch_range_hz"]),
            "pitch_slope_hz_per_sec": float(pitch_profile["pitch_slope_hz_per_sec"]),
            "pitch_std_hz": float(pitch_profile["pitch_std_hz"]),
        }
        logger.info("Source prosody profile: %s", profile)
        return profile

    def map_to_duration(self, input_audio_path, output_audio_path, target_duration_sec, source_prosody=None, transcript_text=""):
        source_prosody = dict(source_prosody or {})
        target_duration = max(0.1, float(target_duration_sec))
        synth_duration = self.ffmpeg_service.get_duration(input_audio_path)
        if synth_duration <= 0:
            raise RuntimeError("Synthesized audio has invalid duration")

        needed_factor = synth_duration / target_duration
        speaking_rate_bias = self._compute_speaking_rate_bias(
            synth_duration=synth_duration,
            transcript_text=transcript_text,
            source_prosody=source_prosody,
        )
        pitch_tension = self._compute_pitch_tension(source_prosody)

        if needed_factor >= 1.0:
            tempo_factor = max(needed_factor, needed_factor * speaking_rate_bias)
            tempo_factor *= 1.004 + max(0.0, (1.0 - pitch_tension)) * 0.006
            if tempo_factor > self.max_tempo_factor:
                logger.warning(
                    "Tempo factor %.3f exceeds configured MAX_TEMPO_FACTOR=%.3f; using %.3f to avoid trimming audio.",
                    tempo_factor,
                    self.max_tempo_factor,
                    tempo_factor,
                )
        else:
            slow_bias = 1.0 / max(0.92, speaking_rate_bias)
            tempo_factor = max(self.min_tempo_factor, needed_factor * slow_bias)
            tempo_factor = min(tempo_factor, 0.999)

        tempo_out = self._temp_with_suffix(output_audio_path, "_tempo.wav")

        factor = float(tempo_factor)
        dur = None
        for _ in range(3):
            self.ffmpeg_service.change_audio_tempo(
                input_audio_path=input_audio_path,
                output_audio_path=tempo_out,
                factor=factor,
            )
            dur = self.ffmpeg_service.get_duration(tempo_out)
            if dur <= target_duration:
                break
            bump = (dur / target_duration) * 1.005
            factor *= bump
        else:
            raise RuntimeError(
                f"Unable to fit audio into target duration without trimming. "
                f"source={synth_duration:.3f}s target={target_duration:.3f}s final={dur:.3f}s"
            )

        self._shape_with_source_prosody(
            input_audio_path=tempo_out,
            output_audio_path=output_audio_path,
            target_duration_sec=target_duration,
            source_prosody=source_prosody,
        )
        Path(tempo_out).unlink(missing_ok=True)
        return output_audio_path

    def _shape_with_source_prosody(self, input_audio_path, output_audio_path, target_duration_sec, source_prosody):
        samples, sample_rate = self._load_audio(input_audio_path)
        target_samples = max(1, int(round(float(target_duration_sec) * sample_rate)))
        samples = self._apply_energy_match(samples, source_prosody)

        if len(samples) > target_samples:
            samples = samples[:target_samples]

        remaining = max(0, target_samples - len(samples))
        leading = self._target_leading_padding_samples(remaining, sample_rate, source_prosody)
        trailing = self._target_trailing_padding_samples(
            remaining=remaining,
            sample_rate=sample_rate,
            source_prosody=source_prosody,
            leading_samples=leading,
        )
        if leading + trailing > remaining:
            trailing = max(0, remaining - leading)
        middle = max(0, remaining - leading - trailing)

        padded = np.concatenate(
            [
                np.zeros(leading, dtype=np.float32),
                samples.astype(np.float32, copy=False),
                np.zeros(middle + trailing, dtype=np.float32),
            ]
        )
        if len(padded) < target_samples:
            padded = np.pad(padded, (0, target_samples - len(padded)))
        elif len(padded) > target_samples:
            padded = padded[:target_samples]

        Path(output_audio_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_audio_path), padded, sample_rate)

    def _compute_speaking_rate_bias(self, synth_duration, transcript_text, source_prosody):
        source_rate = float(source_prosody.get("speaking_rate_wps") or 0.0)
        synth_words = self._count_words(transcript_text)
        if synth_words <= 0:
            synth_words = int(source_prosody.get("word_count") or 0)
        if source_rate <= 0.0 or synth_words <= 0:
            return 1.0
        synth_rate = synth_words / max(0.05, float(synth_duration))
        ratio = synth_rate / max(0.1, source_rate)
        return self._clamp(ratio ** 0.35, 0.92, 1.10)

    def _compute_pitch_tension(self, source_prosody):
        pitch_range = float(source_prosody.get("pitch_range_hz") or 0.0)
        pitch_std = float(source_prosody.get("pitch_std_hz") or 0.0)
        composite = (pitch_range / 220.0) + (pitch_std / 120.0)
        return self._clamp(composite, 0.0, 1.0)

    def _apply_energy_match(self, samples, source_prosody):
        target_db = source_prosody.get("energy_db")
        if target_db is None:
            return samples
        current_db = self._estimate_rms_db(samples)
        gain_db = self._clamp(float(target_db) - float(current_db), -4.0, 4.0)
        gain = 10.0 ** (gain_db / 20.0)
        boosted = samples * gain
        peak = float(np.max(np.abs(boosted))) if boosted.size else 0.0
        if peak > 0.995:
            boosted = boosted / peak * 0.995
        return boosted.astype(np.float32, copy=False)

    def _target_leading_padding_samples(self, remaining, sample_rate, source_prosody):
        if remaining <= 0:
            return 0
        leading_sec = self._clamp(float(source_prosody.get("leading_silence_sec") or 0.0), 0.0, 0.35)
        return min(remaining, int(round(leading_sec * sample_rate)))

    def _target_trailing_padding_samples(self, remaining, sample_rate, source_prosody, leading_samples):
        if remaining <= leading_samples:
            return 0
        trailing_sec = self._clamp(float(source_prosody.get("trailing_silence_sec") or 0.0), 0.0, 0.45)
        slope = float(source_prosody.get("pitch_slope_hz_per_sec") or 0.0)
        pause_ratio = float(source_prosody.get("pause_ratio") or 0.0)
        extra_ratio = 1.0 + self._clamp(slope / 200.0, -0.15, 0.20) + self._clamp(pause_ratio * 0.35, 0.0, 0.15)
        trailing_sec *= extra_ratio
        return min(remaining - leading_samples, int(round(trailing_sec * sample_rate)))

    def _estimate_rms_db(self, samples):
        if samples.size == 0:
            return -60.0
        rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))) + 1e-9)
        return 20.0 * math.log10(max(rms, 1e-9))

    def _estimate_pause_profile(self, samples, sample_rate):
        if samples.size == 0:
            return {
                "pause_ratio": 0.0,
                "pause_count": 0,
                "leading_silence_sec": 0.0,
                "trailing_silence_sec": 0.0,
                "longest_pause_sec": 0.0,
                "total_silence_sec": 0.0,
            }

        frame_len = max(1, int(sample_rate * 0.02))
        hop = max(1, int(sample_rate * 0.01))
        rms_values = self._frame_rms(samples, frame_len, hop)
        if rms_values.size == 0:
            return {
                "pause_ratio": 0.0,
                "pause_count": 0,
                "leading_silence_sec": 0.0,
                "trailing_silence_sec": 0.0,
                "longest_pause_sec": 0.0,
                "total_silence_sec": 0.0,
            }

        silence_threshold = max(np.percentile(rms_values, 25) * 0.55, 5e-4)
        silence_mask = rms_values <= silence_threshold
        silence_runs = self._mask_runs(silence_mask)
        frame_sec = hop / float(sample_rate)
        leading = silence_runs[0][1] * frame_sec if silence_runs and silence_runs[0][0] == 0 else 0.0
        trailing = 0.0
        if silence_runs:
            last_start, last_len = silence_runs[-1]
            if last_start + last_len == len(silence_mask):
                trailing = last_len * frame_sec

        total_silence_sec = float(np.sum(silence_mask) * frame_sec)
        non_edge_runs = []
        for start, length in silence_runs:
            duration = length * frame_sec
            at_edge = start == 0 or (start + length) == len(silence_mask)
            if at_edge:
                continue
            if duration >= 0.08:
                non_edge_runs.append(duration)

        return {
            "pause_ratio": self._clamp(total_silence_sec / max(0.05, len(samples) / float(sample_rate)), 0.0, 0.95),
            "pause_count": len(non_edge_runs),
            "leading_silence_sec": leading,
            "trailing_silence_sec": trailing,
            "longest_pause_sec": max(non_edge_runs) if non_edge_runs else 0.0,
            "total_silence_sec": total_silence_sec,
        }

    def _estimate_pitch_profile(self, samples, sample_rate):
        if samples.size == 0:
            return {
                "pitch_mean_hz": 0.0,
                "pitch_range_hz": 0.0,
                "pitch_slope_hz_per_sec": 0.0,
                "pitch_std_hz": 0.0,
            }

        frame_len = max(1, int(sample_rate * 0.04))
        hop = max(1, int(sample_rate * 0.02))
        min_lag = max(1, int(sample_rate / 350.0))
        max_lag = max(min_lag + 1, int(sample_rate / 75.0))
        pitches = []
        times = []
        for idx, start in enumerate(range(0, max(1, len(samples) - frame_len + 1), hop)):
            frame = samples[start:start + frame_len]
            if frame.size < frame_len:
                break
            if np.sqrt(np.mean(frame * frame) + 1e-9) < 0.01:
                continue
            frame = frame - np.mean(frame)
            corr = np.correlate(frame, frame, mode="full")[frame_len - 1:]
            if corr.size <= max_lag:
                continue
            segment = corr[min_lag:max_lag]
            best_rel = int(np.argmax(segment))
            best_lag = min_lag + best_rel
            peak = float(segment[best_rel])
            if peak <= 0:
                continue
            pitch_hz = float(sample_rate) / float(best_lag)
            if 70.0 <= pitch_hz <= 350.0:
                pitches.append(pitch_hz)
                times.append(idx * hop / float(sample_rate))

        if not pitches:
            return {
                "pitch_mean_hz": 0.0,
                "pitch_range_hz": 0.0,
                "pitch_slope_hz_per_sec": 0.0,
                "pitch_std_hz": 0.0,
            }

        pitch_arr = np.asarray(pitches, dtype=np.float32)
        time_arr = np.asarray(times, dtype=np.float32)
        slope = 0.0
        if pitch_arr.size >= 2 and float(np.ptp(time_arr)) > 0:
            coeffs = np.polyfit(time_arr, pitch_arr, 1)
            slope = float(coeffs[0])
        return {
            "pitch_mean_hz": float(np.mean(pitch_arr)),
            "pitch_range_hz": float(np.max(pitch_arr) - np.min(pitch_arr)),
            "pitch_slope_hz_per_sec": slope,
            "pitch_std_hz": float(np.std(pitch_arr)),
        }

    def _frame_rms(self, samples, frame_len, hop):
        values = []
        for start in range(0, max(1, len(samples) - frame_len + 1), hop):
            frame = samples[start:start + frame_len]
            if frame.size < frame_len:
                break
            values.append(math.sqrt(float(np.mean(frame * frame) + 1e-9)))
        return np.asarray(values, dtype=np.float32)

    def _mask_runs(self, mask):
        runs = []
        start = None
        for idx, value in enumerate(mask):
            if value and start is None:
                start = idx
            elif not value and start is not None:
                runs.append((start, idx - start))
                start = None
        if start is not None:
            runs.append((start, len(mask) - start))
        return runs

    def _load_audio(self, input_audio_path):
        samples, sample_rate = sf.read(str(input_audio_path), always_2d=False)
        if isinstance(samples, np.ndarray) and samples.ndim > 1:
            samples = np.mean(samples, axis=1)
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        return samples, int(sample_rate)

    def _count_words(self, text):
        normalized = " ".join((text or "").split()).strip()
        return len(normalized.split(" ")) if normalized else 0

    def _clamp(self, value, low, high):
        return max(float(low), min(float(high), float(value)))

    def _temp_with_suffix(self, output_audio_path, suffix):
        path = Path(output_audio_path)
        return str(path.with_name(f"{path.stem}{suffix}"))
