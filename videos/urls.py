from django.urls import path
from .views import (
    VideoUploadView,
    UserVideoListView,
    VideoDetailView,
    VideoDeleteView,
    DubbedVideoDownloadView,
)

urlpatterns = [
    path("upload/", VideoUploadView.as_view()),
    path("", UserVideoListView.as_view()),
    path("<int:video_id>/", VideoDetailView.as_view()),
    path("<int:video_id>/delete/", VideoDeleteView.as_view()),
    path("<int:video_id>/download-dubbed/", DubbedVideoDownloadView.as_view()),
]
