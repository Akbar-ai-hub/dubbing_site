import subprocess
import tempfile
from pathlib import Path


class FFmpegService:
    def __init__(self, ffmpeg_bin="ffmpeg"):
        self.ffmpeg_bin = ffmpeg_bin
        self.ffprobe_bin = self._detect_ffprobe(ffmpeg_bin)

    def extract_audio(self, input_video_path, output_audio_path):
        command = [
            self.ffmpeg_bin,
            "-y",
            "-i",
            input_video_path,
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            output_audio_path,
        ]
        self._run(command, "Failed to extract audio with ffmpeg")

    def mux_audio_with_video(self, input_video_path, input_audio_path, output_video_path):
        command = [
            self.ffmpeg_bin,
            "-y",
            "-i",
            input_video_path,
            "-i",
            input_audio_path,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            output_video_path,
        ]
        self._run(command, "Failed to merge dubbed audio with video")

    def extract_audio_segment(self, input_audio_path, output_audio_path, start_sec, end_sec):
        duration = max(0.0, float(end_sec) - float(start_sec))
        if duration <= 0:
            raise RuntimeError("Invalid segment duration for audio extraction")

        command = [
            self.ffmpeg_bin,
            "-y",
            "-ss",
            str(start_sec),
            "-t",
            str(duration),
            "-i",
            input_audio_path,
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            output_audio_path,
        ]
        self._run(command, "Failed to extract audio segment")

    def get_duration(self, input_path):
        command = [
            self.ffprobe_bin,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            input_path,
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            raise RuntimeError(f"Failed to read media duration. {stderr}")
        return float((result.stdout or "0").strip())

    def change_audio_tempo(self, input_audio_path, output_audio_path, factor):
        factor = float(factor)
        if factor <= 0:
            raise RuntimeError("Tempo factor must be positive")

        filters = []
        while factor < 0.5:
            filters.append("atempo=0.5")
            factor /= 0.5
        while factor > 2.0:
            filters.append("atempo=2.0")
            factor /= 2.0
        filters.append(f"atempo={factor:.6f}")
        af = ",".join(filters)

        command = [
            self.ffmpeg_bin,
            "-y",
            "-i",
            input_audio_path,
            "-af",
            af,
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            output_audio_path,
        ]
        self._run(command, "Failed to apply tempo mapping")

    def normalize_audio_duration(self, input_audio_path, output_audio_path, target_duration_sec):
        target = max(0.0, float(target_duration_sec))
        command = [
            self.ffmpeg_bin,
            "-y",
            "-i",
            input_audio_path,
            "-af",
            f"apad=pad_dur={target},atrim=0:{target}",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            output_audio_path,
        ]
        self._run(command, "Failed to normalize audio duration")

    def denoise_audio(self, input_audio_path, output_audio_path, strength_db=-25):
        # Lightweight noise reduction for speaker embedding stability.
        strength_db = float(strength_db)
        command = [
            self.ffmpeg_bin,
            "-y",
            "-i",
            input_audio_path,
            "-af",
            f"afftdn=nf={strength_db}",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            output_audio_path,
        ]
        self._run(command, "Failed to denoise audio")

    def resample_audio(self, input_audio_path, output_audio_path, sample_rate_hz=22050):
        sample_rate_hz = int(sample_rate_hz)
        command = [
            self.ffmpeg_bin,
            "-y",
            "-i",
            input_audio_path,
            "-ar",
            str(sample_rate_hz),
            "-ac",
            "1",
            "-acodec",
            "pcm_s16le",
            output_audio_path,
        ]
        self._run(command, "Failed to resample audio")

    def mix_segments_on_timeline(self, segments, output_audio_path, total_duration_sec):
        if not segments:
            command = [
                self.ffmpeg_bin,
                "-y",
                "-f",
                "lavfi",
                "-i",
                "anullsrc=r=16000:cl=mono",
                "-t",
                str(total_duration_sec),
                output_audio_path,
            ]
            self._run(command, "Failed to create fallback silent audio")
            return

        command = [self.ffmpeg_bin, "-y"]
        for segment in segments:
            command.extend(["-i", segment["path"]])

        filter_parts = []
        labels = []
        for idx, segment in enumerate(segments):
            delay_ms = int(max(0.0, float(segment["start"])) * 1000)
            in_label = f"{idx}:a"
            out_label = f"a{idx}"
            filter_parts.append(f"[{in_label}]adelay={delay_ms}|{delay_ms}[{out_label}]")
            labels.append(f"[{out_label}]")

        mix_label = "".join(labels)
        filter_parts.append(f"{mix_label}amix=inputs={len(segments)}:normalize=0[mixed]")
        filter_complex = ";".join(filter_parts)

        command.extend(
            [
                "-filter_complex",
                filter_complex,
                "-map",
                "[mixed]",
                "-t",
                str(total_duration_sec),
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                "-ac",
                "1",
                output_audio_path,
            ]
        )
        self._run(command, "Failed to mix timeline segments")

    def mix_with_background(self, foreground_audio_path, background_audio_path, output_audio_path, bg_gain_db=-20):
        bg_gain_db = float(bg_gain_db)
        command = [
            self.ffmpeg_bin,
            "-y",
            "-i",
            foreground_audio_path,
            "-i",
            background_audio_path,
            "-filter_complex",
            f"[1:a]volume={bg_gain_db}dB[bg];[0:a][bg]amix=inputs=2:normalize=0[mix]",
            "-map",
            "[mix]",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            output_audio_path,
        ]
        self._run(command, "Failed to mix background audio")

    def concat_audio_files(self, input_audio_paths, output_audio_path):
        paths = [str(p) for p in (input_audio_paths or []) if p]
        if not paths:
            raise RuntimeError("No input audio files provided for concatenation")

        output_path = str(output_audio_path)
        list_file = None
        try:
            with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt", encoding="utf-8") as fp:
                list_file = fp.name
                for p in paths:
                    # concat demuxer requires this format.
                    escaped = p.replace("'", "'\\''")
                    fp.write("file '{}'\n".format(escaped))

            command = [
                self.ffmpeg_bin,
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                list_file,
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                "-ac",
                "1",
                output_path,
            ]
            self._run(command, "Failed to concatenate audio files")
        finally:
            if list_file:
                Path(list_file).unlink(missing_ok=True)

        return output_path

    def _detect_ffprobe(self, ffmpeg_bin):
        normalized = ffmpeg_bin.lower()
        if normalized.endswith("ffmpeg.exe"):
            return ffmpeg_bin[:-10] + "ffprobe.exe"
        if normalized.endswith("ffmpeg"):
            return ffmpeg_bin[:-6] + "ffprobe"
        return "ffprobe"

    def _run(self, command, error_prefix):
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            raise RuntimeError(f"{error_prefix}. {stderr}")
