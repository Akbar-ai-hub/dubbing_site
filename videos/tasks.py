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

        if video.subtitle_srt:
            video.subtitle_srt.delete(save=False)

        video.delete()
        deleted_count += 1

    return {
        "deleted_count": deleted_count,
        "retention_days": retention_days,
    }


@shared_task
def fail_stale_dubbing_jobs():
    timeout_minutes = int(getattr(settings, "DUBBING_STALE_JOB_TIMEOUT_MINUTES", 180))
    cutoff = timezone.now() - timedelta(minutes=max(1, timeout_minutes))
    stale_videos = Video.objects.filter(
        status__in=[Video.STATUS_QUEUED, Video.STATUS_PROCESSING],
        updated_at__lte=cutoff,
    )
    failed_count = stale_videos.update(
        status=Video.STATUS_FAILED,
        progress_percent=0,
        error_message=(
            "Dubbing task did not finish in time. "
            "It may have been interrupted by a worker crash, restart, or timeout."
        ),
        updated_at=timezone.now(),
    )
    return {
        "failed_count": failed_count,
        "timeout_minutes": timeout_minutes,
    }
