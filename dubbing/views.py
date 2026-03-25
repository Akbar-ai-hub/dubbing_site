from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

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

        if video.status == Video.STATUS_PROCESSING:
            return Response(
                {"error": "Dubbing is already in progress"},
                status=status.HTTP_409_CONFLICT,
            )

        video.status = Video.STATUS_PROCESSING
        video.progress_percent = 0
        video.error_message = ""
        video.save(update_fields=["status", "progress_percent", "error_message"])

        try:
            task = process_video_dubbing.delay(video.id)
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
                "message": "Dubbing started",
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
