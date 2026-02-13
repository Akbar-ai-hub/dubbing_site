import subprocess


class FFmpegService:
    def __init__(self, ffmpeg_bin="ffmpeg"):
        self.ffmpeg_bin = ffmpeg_bin

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

    def _run(self, command, error_prefix):
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            raise RuntimeError(f"{error_prefix}. {stderr}")
