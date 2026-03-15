import os
import tempfile

from django.conf import settings
from rest_framework import serializers

from dubbing.services.ffmpeg_service import FFmpegService
from dubbing.services.whisper_service import WhisperService
from .models import Video


class VideoSerializer(serializers.ModelSerializer):
    ALLOWED_ENGLISH_CODES = ("en", "en-us", "en-gb", "english")

    class Meta:
        model = Video
        fields = [
            "id",
            "original_video",
            "dubbed_video",
            "status",
            "error_message",
            "created_at",
        ]
        read_only_fields = ["status", "dubbed_video", "error_message"]

    def validate_original_video(self, file_obj):
        max_mb = int(getattr(settings, "MAX_UPLOAD_VIDEO_MB", 100))
        max_bytes = max_mb * 1024 * 1024
        if file_obj.size > max_bytes:
            raise serializers.ValidationError(f"File size must not exceed {max_mb}MB.")

        allowed_content_types = set(
            getattr(
                settings,
                "ALLOWED_VIDEO_CONTENT_TYPES",
                ["video/mp4", "video/quicktime", "video/x-matroska", "video/webm"],
            )
        )
        content_type = (getattr(file_obj, "content_type", "") or "").lower()
        if content_type and content_type not in allowed_content_types:
            allowed_values = ", ".join(sorted(allowed_content_types))
            raise serializers.ValidationError(
                f"Unsupported content type '{content_type}'. Allowed: {allowed_values}."
            )

        if getattr(settings, "VALIDATE_ORIGINAL_VIDEO_ENGLISH", False):
            self._validate_english_language(file_obj)

        return file_obj

    def _validate_english_language(self, file_obj):
        ffmpeg_bin = getattr(settings, "FFMPEG_BIN", "ffmpeg")
        whisper_model = getattr(settings, "GROQ_ASR_MODEL", "whisper-large-v3")
        groq_api_key = getattr(settings, "GROQ_API_KEY", "")

        if not groq_api_key:
            raise serializers.ValidationError(
                "GROQ_API_KEY must be configured for English language validation."
            )

        source_name = os.path.basename(getattr(file_obj, "name", "") or "upload.mp4")
        suffix = os.path.splitext(source_name)[1] or ".mp4"

        with tempfile.TemporaryDirectory(prefix="video_lang_check_") as tmp_dir:
            video_path = os.path.join(tmp_dir, f"source{suffix}")
            audio_path = os.path.join(tmp_dir, "source.wav")

            with open(video_path, "wb") as output_file:
                for chunk in file_obj.chunks():
                    output_file.write(chunk)

            ffmpeg_service = FFmpegService(ffmpeg_bin=ffmpeg_bin)
            ffmpeg_service.extract_audio(video_path, audio_path)

            whisper_service = WhisperService(
                model_name=whisper_model,
                api_key=groq_api_key,
            )
            transcription = whisper_service.transcribe(audio_path)
            detected_language = (transcription.get("language") or "").strip().lower()

            if not any(
                detected_language == lang or detected_language.startswith(f"{lang}-")
                for lang in self.ALLOWED_ENGLISH_CODES
            ):
                raise serializers.ValidationError(
                    f"Original video language must be English. Detected: '{detected_language or 'unknown'}'."
                )
