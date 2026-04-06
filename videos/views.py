import os
import tempfile
from django.conf import settings
from django.core.files.uploadedfile import SimpleUploadedFile
from django.http import FileResponse
from urllib.parse import urlparse
from rest_framework import status
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import Video, VideoFeedback
from .serializers import VideoSerializer, VideoFeedbackSerializer


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

        return Response(VideoSerializer(video, context={"request": request}).data, status=status.HTTP_201_CREATED)


# ---------------------------
# YOUTUBE DOWNLOAD
# ---------------------------

class YouTubeDownloadView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        url = (request.data.get("url") or "").strip()
        if not url:
            return Response(
                {"error": "YouTube URL was not provided"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        if not self._is_youtube_url(url):
            return Response(
                {"error": "Only YouTube URLs are supported."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            from yt_dlp import YoutubeDL
        except Exception:
            return Response(
                {"error": "yt-dlp is not installed. Install it to enable YouTube downloads."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        max_mb = int(getattr(settings, "MAX_UPLOAD_VIDEO_MB", 100))
        max_bytes = max_mb * 1024 * 1024

        with tempfile.TemporaryDirectory(prefix="yt_download_") as tmp_dir:
            outtmpl = os.path.join(tmp_dir, "%(title)s.%(ext)s")
            ydl_opts = {
                "format": "mp4/bestvideo+bestaudio/best",
                "merge_output_format": "mp4",
                "outtmpl": outtmpl,
                "quiet": True,
                "no_warnings": True,
            }
            try:
                with YoutubeDL(ydl_opts) as ydl:
                    ydl.extract_info(url, download=True)
            except Exception as exc:
                return Response(
                    {"error": f"Failed to download YouTube video: {exc}"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            # Pick the largest file from the temp dir (merged mp4).
            files = [
                os.path.join(tmp_dir, f)
                for f in os.listdir(tmp_dir)
                if os.path.isfile(os.path.join(tmp_dir, f))
            ]
            if not files:
                return Response(
                    {"error": "Downloaded file not found"},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )

            file_path = max(files, key=lambda p: os.path.getsize(p))
            file_size = os.path.getsize(file_path)
            if file_size > max_bytes:
                return Response(
                    {"error": f"Downloaded file exceeds {max_mb}MB limit."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            with open(file_path, "rb") as f:
                content = f.read()

            uploaded = SimpleUploadedFile(
                name=os.path.basename(file_path),
                content=content,
                content_type="video/mp4",
            )

            serializer = VideoSerializer(
                data={"original_video": uploaded},
                context={"request": request},
            )
            serializer.is_valid(raise_exception=True)
            video = serializer.save(
                user=request.user,
                status=Video.STATUS_UPLOADED,
            )

        return Response(VideoSerializer(video, context={"request": request}).data, status=status.HTTP_201_CREATED)

    def _is_youtube_url(self, url):
        try:
            parsed = urlparse(url)
        except Exception:
            return False
        host = (parsed.netloc or "").lower()
        if not host:
            return False
        allowed = {
            "youtube.com",
            "www.youtube.com",
            "m.youtube.com",
            "music.youtube.com",
            "youtu.be",
        }
        return host in allowed


# ---------------------------
# USER VIDEOS LIST
# ---------------------------

class UserVideoListView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        videos = Video.objects.filter(user=request.user).order_by("-created_at")
        serializer = VideoSerializer(videos, many=True, context={"request": request})
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

        serializer = VideoSerializer(video, context={"request": request})
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

        if video.thumbnail:
            video.thumbnail.delete(save=False)

        if video.dubbed_video:
            video.dubbed_video.delete(save=False)

        if video.subtitle_srt:
            video.subtitle_srt.delete(save=False)

        video.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


# ---------------------------
# SHARE LINK CREATE
# ---------------------------

class ShareDubbedVideoView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, video_id):
        try:
            video = Video.objects.get(id=video_id, user=request.user)
        except Video.DoesNotExist:
            return Response({"error": "Video not found"}, status=status.HTTP_404_NOT_FOUND)

        if not video.dubbed_video:
            return Response(
                {"error": "Dubbed video is not available yet"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if not video.share_token:
            video.share_token = self._generate_token()
        video.share_enabled = True
        video.save(update_fields=["share_token", "share_enabled"])

        serializer = VideoSerializer(video, context={"request": request})
        return Response(serializer.data, status=status.HTTP_200_OK)

    def _generate_token(self):
        import uuid

        return uuid.uuid4().hex


# ---------------------------
# SHARE LINK ACCESS (PUBLIC)
# ---------------------------

class SharedDubbedVideoAccessView(APIView):
    permission_classes = [AllowAny]

    def get(self, request, token):
        try:
            video = Video.objects.get(share_token=token, share_enabled=True)
        except Video.DoesNotExist:
            return Response({"error": "Shared video not found"}, status=status.HTTP_404_NOT_FOUND)

        if not video.dubbed_video:
            return Response(
                {"error": "Dubbed video is not available yet"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        file_name = video.dubbed_video.name.split("/")[-1]
        return FileResponse(video.dubbed_video.open("rb"), as_attachment=True, filename=file_name)


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


class VideoFeedbackView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, video_id):
        try:
            video = Video.objects.get(id=video_id, user=request.user)
        except Video.DoesNotExist:
            return Response({"error": "Video not found"}, status=status.HTTP_404_NOT_FOUND)

        if video.status != Video.STATUS_COMPLETED or not video.dubbed_video:
            return Response(
                {"error": "Feedback is available only after dubbing is completed."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        serializer = VideoFeedbackSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        feedback, _ = VideoFeedback.objects.update_or_create(
            video=video,
            user=request.user,
            defaults={
                "rating": serializer.validated_data["rating"],
                "comment": (serializer.validated_data.get("comment") or "").strip(),
            },
        )

        return Response(
            {
                "message": "Feedback saved successfully",
                "video_id": video.id,
                "feedback_rating": feedback.rating,
                "feedback_text": feedback.comment,
                "feedback_updated_at": feedback.updated_at,
            },
            status=status.HTTP_200_OK,
        )
