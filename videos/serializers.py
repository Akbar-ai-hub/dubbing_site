import os
import tempfile
import logging
import math

from django.conf import settings
from django.core.files import File
from rest_framework import serializers

from dubbing.services.ffmpeg_service import FFmpegService
from dubbing.services.whisper_service import WhisperService
from .models import Video, VideoFeedback

logger = logging.getLogger(__name__)


class VideoSerializer(serializers.ModelSerializer):
    ALLOWED_ENGLISH_CODES = ("en", "en-us", "en-gb", "english")
    share_url = serializers.SerializerMethodField()
    feedback_rating = serializers.SerializerMethodField()
    feedback_text = serializers.SerializerMethodField()
    feedback_updated_at = serializers.SerializerMethodField()

    class Meta:
        model = Video
        fields = [
            "id",
            "original_video",
            "duration",
            "thumbnail",
            "dubbed_video",
            "subtitle_srt",
            "share_enabled",
            "share_token",
            "share_url",
            "status",
            "progress_percent",
            "feedback_rating",
            "feedback_text",
            "feedback_updated_at",
            "error_message",
            "created_at",
        ]
        read_only_fields = [
            "status",
            "progress_percent",
            "duration",
            "thumbnail",
            "feedback_rating",
            "feedback_text",
            "feedback_updated_at",
            "dubbed_video",
            "subtitle_srt",
            "share_enabled",
            "share_token",
            "share_url",
            "error_message",
        ]

    def create(self, validated_data):
        video = super().create(validated_data)
        self._populate_video_metadata(video)
        return video

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

    def _populate_video_metadata(self, video):
        video_path = getattr(video.original_video, "path", None)
        if not video_path or not os.path.exists(video_path):
            return

        ffmpeg_bin = getattr(settings, "FFMPEG_BIN", "ffmpeg")
        ffmpeg_service = FFmpegService(ffmpeg_bin=ffmpeg_bin)
        update_fields = []
        duration = None

        try:
            duration = float(ffmpeg_service.get_duration(video_path))
            if not math.isfinite(duration) or duration <= 0:
                duration = None
        except Exception as exc:
            logger.warning("Failed to detect video duration for Video #%s: %s", video.id, exc)

        if duration is not None:
            video.duration = round(duration, 3)
            update_fields.append("duration")

        try:
            with tempfile.TemporaryDirectory(prefix=f"video_thumb_{video.id}_") as tmp_dir:
                thumbnail_path = os.path.join(tmp_dir, "thumbnail.jpg")
                at_sec = 0.0
                if duration and duration > 1:
                    at_sec = min(duration * 0.1, 5.0)

                ffmpeg_service.extract_thumbnail(
                    input_video_path=video_path,
                    output_image_path=thumbnail_path,
                    at_sec=at_sec,
                )

                if os.path.exists(thumbnail_path) and os.path.getsize(thumbnail_path) > 0:
                    with open(thumbnail_path, "rb") as fp:
                        video.thumbnail.save(f"video_{video.id}_thumb.jpg", File(fp), save=False)
                    update_fields.append("thumbnail")
        except Exception as exc:
            logger.warning("Failed to generate thumbnail for Video #%s: %s", video.id, exc)

        if update_fields:
            video.save(update_fields=list(dict.fromkeys(update_fields)))

    def get_share_url(self, obj):
        request = self.context.get("request")
        if not obj.share_enabled or not obj.share_token or not request:
            return None
        return request.build_absolute_uri(f"/api/videos/share/{obj.share_token}/")

    def _get_feedback(self, obj):
        return VideoFeedback.objects.filter(video=obj, user=obj.user).order_by("-updated_at").first()

    def get_feedback_rating(self, obj):
        feedback = self._get_feedback(obj)
        return feedback.rating if feedback else None

    def get_feedback_text(self, obj):
        feedback = self._get_feedback(obj)
        return feedback.comment if feedback else ""

    def get_feedback_updated_at(self, obj):
        feedback = self._get_feedback(obj)
        return feedback.updated_at if feedback else None


class VideoFeedbackSerializer(serializers.Serializer):
    rating = serializers.IntegerField(min_value=1, max_value=5)
    comment = serializers.CharField(required=False, allow_blank=True, max_length=2000)


class VideoFeedbackListItemSerializer(serializers.ModelSerializer):
    video_id = serializers.IntegerField(source="video.id", read_only=True)
    video_status = serializers.CharField(source="video.status", read_only=True)
    username = serializers.CharField(source="user.username", read_only=True)

    class Meta:
        model = VideoFeedback
        fields = [
            "id",
            "video_id",
            "video_status",
            "username",
            "rating",
            "comment",
            "created_at",
            "updated_at",
        ]
