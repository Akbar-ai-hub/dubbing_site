from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
from django.conf import settings
from decimal import Decimal

from videos.models import Video
from videos.serializers import VideoSerializer

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
            min_balance = Decimal(str(getattr(settings, "BILLING_MIN_START_BALANCE", "0.10")))
            user_balance = Decimal(str(request.user.balance))
            if user_balance < min_balance:
                return Response(
                    {"error": f"Insufficient balance. Minimum required to start dubbing is {min_balance}."},
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


class DubbingStatusView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, video_id):
        try:
            video = Video.objects.get(id=video_id, user=request.user)
        except Video.DoesNotExist:
            return Response({"error": "Video not found"}, status=status.HTTP_404_NOT_FOUND)

        serializer = VideoSerializer(video)
        return Response(serializer.data, status=status.HTTP_200_OK)
