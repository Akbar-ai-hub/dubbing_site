from django.http import FileResponse
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import Video
from .serializers import VideoSerializer


# ---------------------------
# VIDEO UPLOAD
# ---------------------------

class VideoUploadView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        file = request.FILES.get("video")

        if not file:
            return Response(
                {"error": "Video file was not sent"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        serializer = VideoSerializer(data={"original_video": file}, context={"request": request})
        serializer.is_valid(raise_exception=True)

        video = serializer.save(
            user=request.user,
            status=Video.STATUS_UPLOADED,
        )

        return Response(VideoSerializer(video).data, status=status.HTTP_201_CREATED)


# ---------------------------
# USER VIDEOS LIST
# ---------------------------

class UserVideoListView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        videos = Video.objects.filter(user=request.user).order_by("-created_at")
        serializer = VideoSerializer(videos, many=True)
        return Response(serializer.data)


# ---------------------------
# VIDEO DETAIL
# ---------------------------

class VideoDetailView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, video_id):
        try:
            video = Video.objects.get(id=video_id, user=request.user)
        except Video.DoesNotExist:
            return Response({"error": "Video not found"}, status=status.HTTP_404_NOT_FOUND)

        serializer = VideoSerializer(video)
        return Response(serializer.data)


# ---------------------------
# VIDEO DELETE
# ---------------------------

class VideoDeleteView(APIView):
    permission_classes = [IsAuthenticated]

    def delete(self, request, video_id):
        try:
            video = Video.objects.get(id=video_id, user=request.user)
        except Video.DoesNotExist:
            return Response({"error": "Video not found"}, status=status.HTTP_404_NOT_FOUND)

        if video.original_video:
            video.original_video.delete(save=False)

        if video.dubbed_video:
            video.dubbed_video.delete(save=False)

        if video.subtitle_srt:
            video.subtitle_srt.delete(save=False)

        video.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


# ---------------------------
# DUBBED VIDEO DOWNLOAD
# ---------------------------

class DubbedVideoDownloadView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, video_id):
        try:
            video = Video.objects.get(id=video_id, user=request.user)
        except Video.DoesNotExist:
            return Response({"error": "Video not found"}, status=status.HTTP_404_NOT_FOUND)

        if not video.dubbed_video:
            return Response(
                {"error": "Dubbed video is not available yet"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        file_name = video.dubbed_video.name.split("/")[-1]
        return FileResponse(video.dubbed_video.open("rb"), as_attachment=True, filename=file_name)
