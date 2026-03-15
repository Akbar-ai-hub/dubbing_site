from pathlib import Path
import logging


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

    def map_to_duration(self, input_audio_path, output_audio_path, target_duration_sec):
        target_duration = max(0.1, float(target_duration_sec))
        source_duration = self.ffmpeg_service.get_duration(input_audio_path)
        if source_duration <= 0:
            raise RuntimeError("Synthesized audio has invalid duration")

        # `tempo_factor` is the ffmpeg atempo factor. >1 speeds up (shorter), <1 slows down (longer).
        #
        # Requirements:
        # 1) Segment audio must fit in the target window without cutting content.
        #    If it doesn't fit, increase tempo until it fits.
        # 2) Avoid excessive slow-down (min_tempo_factor is a lower bound).
        needed_factor = source_duration / target_duration
        if needed_factor >= 1.0:
            # Speed-up as much as needed to fit. max_tempo_factor becomes a soft cap only.
            if needed_factor > self.max_tempo_factor:
                logger.warning(
                    "Tempo factor %.3f exceeds configured MAX_TEMPO_FACTOR=%.3f; using %.3f to avoid trimming audio.",
                    needed_factor,
                    self.max_tempo_factor,
                    needed_factor,
                )
            # Safety margin to reduce chance of being slightly longer due to rounding.
            tempo_factor = float(needed_factor) * 1.01
        else:
            # Slow-down to fill the window, but not below the configured minimum.
            tempo_factor = max(self.min_tempo_factor, float(needed_factor))
        tempo_out = self._temp_with_suffix(output_audio_path, "_tempo.wav")

        # Iteratively speed-up if still too long; never trim content to fit.
        factor = float(tempo_factor)
        dur = None
        for _ in range(3):
            self.ffmpeg_service.change_audio_tempo(
                input_audio_path=input_audio_path,
                output_audio_path=tempo_out,
                factor=factor,
            )
            dur = self.ffmpeg_service.get_duration(tempo_out)
            # Must be <= target_duration so that the next step only trims silence (not content).
            if dur <= target_duration:
                break
            bump = (dur / target_duration) * 1.01
            factor *= bump
        else:
            raise RuntimeError(
                f"Unable to fit audio into target duration without trimming. "
                f"source={source_duration:.3f}s target={target_duration:.3f}s final={dur:.3f}s"
            )

        self.ffmpeg_service.normalize_audio_duration(
            input_audio_path=tempo_out,
            output_audio_path=output_audio_path,
            target_duration_sec=target_duration,
        )
        Path(tempo_out).unlink(missing_ok=True)
        return output_audio_path

    def _temp_with_suffix(self, output_audio_path, suffix):
        path = Path(output_audio_path)
        return str(path.with_name(f"{path.stem}{suffix}"))
