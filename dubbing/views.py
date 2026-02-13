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
        video.error_message = ""
        video.save(update_fields=["status", "error_message"])

        task = process_video_dubbing.delay(video.id)
        return Response(
            {
                "message": "Dubbing started",
                "task_id": task.id,
                "video_id": video.id,
                "status": video.status,
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
