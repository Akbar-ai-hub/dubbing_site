from django.db import models
from django.conf import settings

User = settings.AUTH_USER_MODEL


class Video(models.Model):
    STATUS_UPLOADED = "uploaded"
    STATUS_QUEUED = "queued"
    STATUS_PROCESSING = "processing"
    STATUS_COMPLETED = "completed"
    STATUS_FAILED = "failed"

    STATUS_CHOICES = [
        (STATUS_UPLOADED, "Uploaded"),
        (STATUS_QUEUED, "Queued"),
        (STATUS_PROCESSING, "Processing"),
        (STATUS_COMPLETED, "Completed"),
        (STATUS_FAILED, "Failed"),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="videos")
    original_video = models.FileField(upload_to="original_videos/")
    duration = models.FloatField(blank=True, null=True)
    thumbnail = models.FileField(upload_to="video_thumbnails/", blank=True, null=True)
    dubbed_video = models.FileField(upload_to="dubbed_videos/", blank=True, null=True)
    subtitle_srt = models.FileField(upload_to="dubbed_videos/", blank=True, null=True)
    share_enabled = models.BooleanField(default=False)
    share_token = models.CharField(max_length=64, blank=True, null=True, unique=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default=STATUS_UPLOADED)
    progress_percent = models.PositiveSmallIntegerField(default=0)
    error_message = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Video #{self.id} ({self.status})"


class VideoFeedback(models.Model):
    video = models.ForeignKey(Video, on_delete=models.CASCADE, related_name="feedbacks")
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="video_feedbacks")
    rating = models.PositiveSmallIntegerField()
    comment = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["video", "user"], name="uniq_video_feedback_per_user"),
        ]

    def __str__(self):
        return f"Feedback(video={self.video_id}, user={self.user_id}, rating={self.rating})"

