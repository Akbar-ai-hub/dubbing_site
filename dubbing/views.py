from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
from django.conf import settings
from decimal import Decimal, ROUND_HALF_UP

from videos.models import Video
from videos.serializers import VideoSerializer
from dubbing.services.ffmpeg_service import FFmpegService

from .tasks import process_video_dubbing


class StartDubbingView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, video_id):
        try:
            video = Video.objects.get(id=video_id, user=request.user)
        except Video.DoesNotExist:
            return Response({"error": "Video not found"}, status=status.HTTP_404_NOT_FOUND)

        if not video.original_video:
            return Response(
                {"error": "Original video is missing"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if video.status in (Video.STATUS_QUEUED, Video.STATUS_PROCESSING):
            return Response(
                {"error": "Dubbing is already in progress or queued"},
                status=status.HTTP_409_CONFLICT,
            )

        per_user_limit = int(getattr(settings, "DUBBING_MAX_ACTIVE_PER_USER", 1))
        global_limit = int(getattr(settings, "DUBBING_MAX_GLOBAL_ACTIVE", 20))
        user_active = Video.objects.filter(
            user=request.user,
            status__in=[Video.STATUS_QUEUED, Video.STATUS_PROCESSING],
        ).count()
        if user_active >= per_user_limit:
            return Response(
                {"error": "You already have an active dubbing task. Please wait until it finishes."},
                status=status.HTTP_429_TOO_MANY_REQUESTS,
            )

        global_active = Video.objects.filter(
            status__in=[Video.STATUS_QUEUED, Video.STATUS_PROCESSING],
        ).count()
        if global_active >= global_limit:
            return Response(
                {"error": "Dubbing queue is temporarily full. Please try again shortly."},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        billing_enabled = str(getattr(settings, "BILLING_ENABLED", "true")).strip().lower() in {"1", "true", "yes", "on"}
        if billing_enabled:
            billing_currency = str(getattr(settings, "BILLING_CURRENCY", "KZT")).upper()
            user_balance = Decimal(str(request.user.balance))
            try:
                required_amount = self._estimate_required_balance(video)
            except Exception as exc:
                return Response(
                    {"error": f"Failed to estimate dubbing cost: {exc}"},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            if user_balance < required_amount:
                return Response(
                    {
                        "error": (
                            "Insufficient balance for this video. "
                            f"Estimated dubbing cost with safety margin is {required_amount} {billing_currency}, "
                            f"but your balance is {user_balance} {billing_currency}."
                        )
                    },
                    status=status.HTTP_402_PAYMENT_REQUIRED,
                )

        video.status = Video.STATUS_QUEUED
        video.progress_percent = 0
        video.error_message = ""
        video.save(update_fields=["status", "progress_percent", "error_message"])

        try:
            task = process_video_dubbing.apply_async(
                args=[video.id],
                queue=getattr(settings, "CELERY_DUBBING_GPU_QUEUE", "dubbing_gpu"),
            )
        except Exception as exc:
            video.status = Video.STATUS_FAILED
            video.error_message = f"Failed to enqueue dubbing task: {exc}"
            video.save(update_fields=["status", "error_message"])
            return Response(
                {"error": "Failed to start dubbing task"},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        return Response(
            {
                "message": "Dubbing queued",
                "task_id": task.id,
                "video_id": video.id,
                "status": video.status,
                "progress_percent": video.progress_percent,
            },
            status=status.HTTP_202_ACCEPTED,
        )

    def _estimate_required_balance(self, video):
        duration_sec = self._get_video_duration(video)
        estimate_price_per_video_min = self._to_decimal(
            getattr(settings, "DUBBING_ESTIMATE_PRICE_PER_VIDEO_MINUTE", "2000.00"),
            "2000.00",
        )
        duration_minutes = Decimal(str(max(0.0, float(duration_sec)))) / Decimal("60")
        base_cost = duration_minutes * estimate_price_per_video_min
        safety_multiplier = Decimal("1.20")
        return (base_cost * safety_multiplier).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    def _get_video_duration(self, video):
        if video.duration and float(video.duration) > 0:
            return float(video.duration)

        video_path = getattr(video.original_video, "path", None)
        if not video_path:
            raise RuntimeError("Video duration could not be determined.")

        ffmpeg_service = FFmpegService(ffmpeg_bin=getattr(settings, "FFMPEG_BIN", "ffmpeg"))
        duration = float(ffmpeg_service.get_duration(video_path))
        if duration <= 0:
            raise RuntimeError("Video duration could not be determined.")

        video.duration = round(duration, 3)
        video.save(update_fields=["duration"])
        return duration

    def _to_decimal(self, value, default="0"):
        try:
            return Decimal(str(value))
        except Exception:
            return Decimal(str(default))


class DubbingStatusView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, video_id):
        try:
            video = Video.objects.get(id=video_id, user=request.user)
        except Video.DoesNotExist:
            return Response({"error": "Video not found"}, status=status.HTTP_404_NOT_FOUND)

        serializer = VideoSerializer(video)
        return Response(serializer.data, status=status.HTTP_200_OK)
