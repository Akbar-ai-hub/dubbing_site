from django.urls import path
from .views import (
    VideoUploadView,
    YouTubeDownloadView,
    UserVideoListView,
    VideoDetailView,
    VideoDeleteView,
    DubbedVideoDownloadView,
    ShareDubbedVideoView,
    SharedDubbedVideoAccessView,
    VideoFeedbackView,
    VideoFeedbackListView,
)

urlpatterns = [
    path("upload/", VideoUploadView.as_view()),
    path("youtube/", YouTubeDownloadView.as_view()),
    path("", UserVideoListView.as_view()),
    path("<int:video_id>/", VideoDetailView.as_view()),
    path("<int:video_id>/delete/", VideoDeleteView.as_view()),
    path("<int:video_id>/share/", ShareDubbedVideoView.as_view()),
    path("<int:video_id>/feedback/", VideoFeedbackView.as_view()),
    path("feedbacks/", VideoFeedbackListView.as_view()),
    path("share/<str:token>/", SharedDubbedVideoAccessView.as_view()),
    path("<int:video_id>/download-dubbed/", DubbedVideoDownloadView.as_view()),
]
