from datetime import timedelta

from celery import shared_task
from django.conf import settings
from django.utils import timezone

from .models import Video


@shared_task
def delete_expired_videos():
    retention_days = getattr(settings, "VIDEO_RETENTION_DAYS", 7)
    cutoff = timezone.now() - timedelta(days=retention_days)

    expired_videos = Video.objects.filter(created_at__lte=cutoff)
    deleted_count = 0

    for video in expired_videos.iterator():
        if video.original_video:
            video.original_video.delete(save=False)

        if video.dubbed_video:
            video.dubbed_video.delete(save=False)

        video.delete()
        deleted_count += 1

    return {
        "deleted_count": deleted_count,
        "retention_days": retention_days,
    }
