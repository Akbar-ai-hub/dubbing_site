from django.urls import path
from .views import StartDubbingView, DubbingStatusView

urlpatterns = [
    path("<int:video_id>/start/", StartDubbingView.as_view(), name="dubbing-start"),
    path("<int:video_id>/status/", DubbingStatusView.as_view(), name="dubbing-status"),
]
