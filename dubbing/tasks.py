import os
import shutil
import tempfile
import time
from pathlib import Path
import logging
from decimal import Decimal, ROUND_HALF_UP

from celery import shared_task
from django.conf import settings
from django.core.files.base import File
from django.core.mail import send_mail
from django.db import transaction

from videos.models import Video
from users.models import BillingTransaction, User, NotificationPreference, UserNotification

from .services import DubbingPipelineService


def _get_setting(name, default=None):
    return getattr(settings, name, os.environ.get(name, default))


def _to_decimal(value, default="0"):
    try:
        return Decimal(str(value))
    except Exception:
        return Decimal(str(default))


def _bill_video_usage(user_id, gpu_seconds, video_id=None, billing_reason="completed"):
    billing_enabled = str(getattr(settings, "BILLING_ENABLED", "true")).strip().lower() in {"1", "true", "yes", "on"}
    if not billing_enabled:
        return Decimal("0.00")

    currency = str(getattr(settings, "BILLING_CURRENCY", "KZT")).upper()
    gpu_price_per_min = _to_decimal(getattr(settings, "GPU_PRICE_PER_MINUTE", "15.00"), "15.00")
    storage_price_per_gb = _to_decimal(getattr(settings, "STORAGE_PRICE_PER_GB", "0.50"), "0.50")

    gpu_minutes = (Decimal(str(max(0.0, gpu_seconds))) / Decimal("60"))
    gpu_cost = (gpu_minutes * gpu_price_per_min)

    video_obj = None
    if video_id:
        try:
            video_obj = Video.objects.get(id=video_id)
        except Video.DoesNotExist:
            video_obj = None

    total_bytes = 0
    if video_obj is not None:
        for f in [video_obj.original_video, video_obj.dubbed_video, video_obj.subtitle_srt]:
            if not f:
                continue
            try:
                total_bytes += int(f.size)
            except Exception:
                pass
    storage_gb = (Decimal(str(total_bytes)) / Decimal(str(1024 ** 3)))
    storage_cost = storage_gb * storage_price_per_gb

    total_cost = (gpu_cost + storage_cost).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    if total_cost <= Decimal("0.00"):
        return Decimal("0.00")

    with transaction.atomic():
        user = User.objects.select_for_update().get(id=user_id)
        if Decimal(str(user.balance)) < total_cost:
            raise RuntimeError(
                f"Insufficient balance for billing. Required={total_cost}, available={user.balance}."
            )
        user.balance = (Decimal(str(user.balance)) - total_cost).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        user.save(update_fields=["balance"])
        BillingTransaction.objects.create(
            user=user,
            video=video_obj,
            txn_type=BillingTransaction.TYPE_DUBBING_CHARGE,
            amount=total_cost,
            description=(
                f"Dubbing charge ({currency}, status={billing_reason}): "
                f"gpu_minutes={gpu_minutes.quantize(Decimal('0.001'))}, "
                f"gpu_cost={gpu_cost.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)}, "
                f"storage_gb={storage_gb.quantize(Decimal('0.0001'))}, "
                f"storage_cost={storage_cost.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)}"
            ),
        )
    return total_cost


def _update_video_progress(video_id, percent, stage=None):
    clamped = max(0, min(100, int(percent)))
    Video.objects.filter(id=video_id).update(progress_percent=clamped)
    if stage:
        logger.info("Dubbing progress video=%s %s%% stage=%s", video_id, clamped, stage)


def _create_dubbing_notification(user_id, video_id, is_success, error_message=""):
    try:
        user = User.objects.get(id=user_id)
        preferences, _ = NotificationPreference.objects.get_or_create(user_id=user_id)
        if not preferences.notify_completed:
            return

        if is_success:
            title = "Dubbing completed"
            message = f"Your video #{video_id} has been dubbed successfully."
            notification_type = UserNotification.TYPE_DUBBING_COMPLETED
        else:
            title = "Dubbing failed"
            suffix = f" Error: {error_message}" if error_message else ""
            message = f"Your video #{video_id} dubbing failed.{suffix}"
            notification_type = UserNotification.TYPE_DUBBING_FAILED

        UserNotification.objects.create(
            user_id=user_id,
            video_id=video_id,
            title=title,
            message=message,
            notification_type=notification_type,
        )

        if preferences.notify_email and user.email:
            subject = title
            email_message = (
                f"Hello {user.username},\n\n"
                f"{message}\n\n"
                "Video Dubbing Service"
            )
            send_mail(
                subject=subject,
                message=email_message,
                from_email=getattr(settings, "EMAIL_HOST_USER", None),
                recipient_list=[user.email],
                fail_silently=True,
            )
    except Exception as exc:
        logger.warning("Failed to create user notification for video=%s: %s", video_id, exc)


def _cleanup_stale_subtitles(video_id, keep_name=None):
    subtitle_dir = Path(settings.MEDIA_ROOT) / "dubbed_videos"
    if not subtitle_dir.exists():
        return

    keep_basename = Path(keep_name).name if keep_name else None
    pattern = f"dubbed_{video_id}_*.srt"
    for path in subtitle_dir.glob(pattern):
        if keep_basename and path.name == keep_basename:
            continue
        try:
            path.unlink()
            logger.info("Deleted stale subtitle file: %s", path)
        except FileNotFoundError:
            continue
        except Exception as exc:
            logger.warning("Failed to delete stale subtitle file %s: %s", path, exc)

logger = logging.getLogger(__name__)

@shared_task
def process_video_dubbing(video_id):
    try:
        video = Video.objects.get(id=video_id)
    except Video.DoesNotExist:
        return {"error": "Video not found"}

    if video.status not in (Video.STATUS_QUEUED, Video.STATUS_PROCESSING):
        return {"video_id": video_id, "status": "skipped", "reason": f"status={video.status}"}

    if video.status != Video.STATUS_PROCESSING:
        video.status = Video.STATUS_PROCESSING
        video.progress_percent = 1
        video.error_message = ""
        video.save(update_fields=["status", "progress_percent", "error_message"])

    if not video.original_video:
        video.status = Video.STATUS_FAILED
        video.progress_percent = 0
        video.error_message = "Original video is missing"
        video.save(update_fields=["status", "progress_percent", "error_message"])
        return {"error": "Original video is missing"}

    user_id = video.user_id
    billing_started = None
    billing_reason = "completed"
    result = {}

    try:
        _update_video_progress(video.id, 2, "initializing")
        original_name = Path(video.original_video.name).name
        ffmpeg_bin = _get_setting("FFMPEG_BIN", "ffmpeg")
        whisper_model = _get_setting("GROQ_ASR_MODEL", "whisper-large-v3")
        groq_api_key = _get_setting("GROQ_API_KEY", "")
        nllb_model_dir = _get_setting(
            "NLLB_MODEL_DIR",
            r"C:\Users\AKBAR\.cache\huggingface\hub\models--facebook--nllb-200-distilled-600M",
        )
        nllb_source_lang = _get_setting("NLLB_SOURCE_LANG", "eng_Latn")
        nllb_target_lang = _get_setting("NLLB_TARGET_LANG", "kaz_Cyrl")
        nllb_batch_size = int(_get_setting("NLLB_BATCH_SIZE", 8))
        nllb_max_new_tokens = int(_get_setting("NLLB_MAX_NEW_TOKENS", 256))
        source_language_name = _get_setting("DUBBING_SOURCE_LANGUAGE_NAME", "English")
        target_language_name = _get_setting("DUBBING_TARGET_LANGUAGE_NAME", "Kazakh")
        diarization_model = _get_setting(
            "DIARIZATION_MODEL_NAME", "pyannote/speaker-diarization-3.1"
        )
        diarization_min_speakers = _get_setting("DIARIZATION_MIN_SPEAKERS", "")
        diarization_max_speakers = _get_setting("DIARIZATION_MAX_SPEAKERS", "")
        huggingface_token = _get_setting("HUGGINGFACE_TOKEN", "")
        tts_target_language = _get_setting("TTS_TARGET_LANGUAGE", "")
        tts_xtts_local_dir = _get_setting("COQUI_XTTS_LOCAL_DIR", "")
        xtts_temperature = float(_get_setting("XTTS_TEMPERATURE", 0.5))
        xtts_length_penalty = float(_get_setting("XTTS_LENGTH_PENALTY", 1.0))
        xtts_repetition_penalty = float(_get_setting("XTTS_REPETITION_PENALTY", 3.0))
        xtts_top_k = int(_get_setting("XTTS_TOP_K", 50))
        xtts_top_p = float(_get_setting("XTTS_TOP_P", 0.9))
        min_segment_chars = int(_get_setting("MIN_SEGMENT_CHARS", 35))
        max_merge_chars = int(_get_setting("MAX_MERGE_CHARS", 220))
        speaker_embedding_model_name = _get_setting("SPEAKER_EMBEDDING_MODEL_NAME", "pyannote/embedding")
        # Backward-compat: if SPEAKER_EMBEDDING_MODEL_DIR is set to ...\\hub\\models--...,
        # we can infer the hub cache directory (the parent of models--...).
        speaker_embedding_cache_dir = _get_setting("SPEAKER_EMBEDDING_CACHE_DIR", "")
        legacy_model_dir = _get_setting("SPEAKER_EMBEDDING_MODEL_DIR", "")
        if not speaker_embedding_cache_dir and legacy_model_dir:
            try:
                p = Path(legacy_model_dir).expanduser().resolve()
                if p.name.startswith("models--") and p.parent.name.lower() == "hub":
                    speaker_embedding_cache_dir = str(p.parent)
                elif p.name.lower() == "hub":
                    speaker_embedding_cache_dir = str(p)
            except Exception:
                pass
        speaker_embedding_device = _get_setting("SPEAKER_EMBEDDING_DEVICE", "")
        source_language = _get_setting("DUBBING_SOURCE_LANGUAGE", None)

        subtitle_basename = f"dubbed_{video.id}_{Path(original_name).stem}.srt"

        pipeline = DubbingPipelineService(
            ffmpeg_bin=ffmpeg_bin,
            whisper_model_name=whisper_model,
            translation_model_dir=nllb_model_dir,
            groq_api_key=groq_api_key,
            source_language_name=source_language_name,
            target_language_name=target_language_name,
            nllb_source_lang=nllb_source_lang,
            nllb_target_lang=nllb_target_lang,
            nllb_batch_size=nllb_batch_size,
            nllb_max_new_tokens=nllb_max_new_tokens,
            source_language=source_language,
            diarization_model_name=diarization_model,
            hf_auth_token=huggingface_token,
            diarization_min_speakers=diarization_min_speakers,
            diarization_max_speakers=diarization_max_speakers,
            tts_xtts_local_dir=(tts_xtts_local_dir or None),
            tts_target_language=tts_target_language,
            xtts_temperature=xtts_temperature,
            xtts_length_penalty=xtts_length_penalty,
            xtts_repetition_penalty=xtts_repetition_penalty,
            xtts_top_k=xtts_top_k,
            xtts_top_p=xtts_top_p,
            min_segment_chars=min_segment_chars,
            max_merge_chars=max_merge_chars,
            speaker_embedding_model_name=speaker_embedding_model_name,
            speaker_embedding_cache_dir=(speaker_embedding_cache_dir or None),
            speaker_embedding_model_dir=(legacy_model_dir or None),
            speaker_embedding_device=(speaker_embedding_device or None),
            ref_audio_output_dir=str(Path(settings.MEDIA_ROOT) / "dubbed_videos"),
            subtitle_basename=subtitle_basename,
        )

        source_suffix = Path(original_name).suffix or ".mp4"
        old_dubbed_name = video.dubbed_video.name if video.dubbed_video else None
        old_subtitle_name = video.subtitle_srt.name if video.subtitle_srt else None

        with tempfile.TemporaryDirectory(prefix=f"dubbing_{video.id}_") as temp_dir:
            input_video_path = str(Path(temp_dir) / f"input{source_suffix}")
            extracted_audio_path = str(Path(temp_dir) / "extracted.wav")
            tts_audio_path = str(Path(temp_dir) / "tts.wav")
            output_video_path = str(Path(temp_dir) / f"output{source_suffix}")

            with video.original_video.open("rb") as source_file, open(input_video_path, "wb") as dst:
                shutil.copyfileobj(source_file, dst)

            billing_started = time.monotonic()
            result = pipeline.run(
                input_video_path=input_video_path,
                extracted_audio_path=extracted_audio_path,
                tts_audio_path=tts_audio_path,
                output_video_path=output_video_path,
                progress_callback=lambda percent, stage: _update_video_progress(video.id, percent, stage),
            )

            logger.info("Whisper transcript: %s", result.get("transcript_text"))
            logger.info("Translated text: %s", result.get("translated_text"))


            output_name = f"dubbed_{video.id}_{Path(original_name).stem}{source_suffix}"
            with open(output_video_path, "rb") as output_file:
                video.dubbed_video.save(output_name, File(output_file), save=False)

            subtitle_path = Path(settings.MEDIA_ROOT) / "dubbed_videos" / subtitle_basename
            if subtitle_path.exists():
                video.subtitle_srt.name = f"dubbed_videos/{subtitle_basename}"

        if not Video.objects.filter(id=video.id).exists():
            raise RuntimeError("Video was deleted during processing.")

        video.status = Video.STATUS_COMPLETED
        video.progress_percent = 100
        video.error_message = ""
        video.save(update_fields=["dubbed_video", "subtitle_srt", "status", "progress_percent", "error_message"])

        _cleanup_stale_subtitles(video.id, keep_name=video.subtitle_srt.name if video.subtitle_srt else None)

        if old_dubbed_name and old_dubbed_name != video.dubbed_video.name:
            video.dubbed_video.storage.delete(old_dubbed_name)
        if old_subtitle_name and video.subtitle_srt and old_subtitle_name != video.subtitle_srt.name:
            video.subtitle_srt.storage.delete(old_subtitle_name)

        _create_dubbing_notification(user_id=user_id, video_id=video.id, is_success=True)

        return {
            "video_id": video.id,
            "status": video.status,
            "detected_language": result.get("detected_language"),
            "segments_count": result.get("segments_count"),
        }
    except Exception as exc:
        error_text = str(exc)
        billing_reason = "failed"
        Video.objects.filter(id=video.id).update(
            status=Video.STATUS_FAILED,
            progress_percent=0,
            error_message=error_text,
        )
        _create_dubbing_notification(user_id=user_id, video_id=video.id, is_success=False, error_message=error_text)
        return {"video_id": video.id, "status": Video.STATUS_FAILED, "error": error_text}
    finally:
        if billing_started is not None:
            billing_elapsed = max(0.0, time.monotonic() - billing_started)
            try:
                charged_amount = _bill_video_usage(
                    user_id=user_id,
                    gpu_seconds=billing_elapsed,
                    video_id=video_id,
                    billing_reason=billing_reason,
                )
                logger.info(
                    "Billing charged for video=%s amount=%s reason=%s elapsed_sec=%.3f",
                    video_id,
                    charged_amount,
                    billing_reason,
                    billing_elapsed,
                )
            except Exception as bill_exc:
                logger.exception(
                    "Billing failed for video=%s reason=%s elapsed_sec=%.3f: %s",
                    video_id,
                    billing_reason,
                    billing_elapsed,
                    bill_exc,
                )
