import os
import shutil
import tempfile
from pathlib import Path
import logging

from celery import shared_task
from django.conf import settings
from django.core.files.base import File
from pathlib import Path

from videos.models import Video

from .services import DubbingPipelineService


def _get_setting(name, default=None):
    return getattr(settings, name, os.environ.get(name, default))

logger = logging.getLogger(__name__)

@shared_task
def process_video_dubbing(video_id):
    try:
        video = Video.objects.get(id=video_id)
    except Video.DoesNotExist:
        return {"error": "Video not found"}

    if not video.original_video:
        video.status = Video.STATUS_FAILED
        video.error_message = "Original video is missing"
        video.save(update_fields=["status", "error_message"])
        return {"error": "Original video is missing"}

    try:
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
        xtts_fallback_language = _get_setting("XTTS_FALLBACK_LANGUAGE", "tr")
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
            xtts_fallback_language=xtts_fallback_language,
            tts_target_language=tts_target_language,
            speaker_embedding_model_name=speaker_embedding_model_name,
            speaker_embedding_cache_dir=(speaker_embedding_cache_dir or None),
            speaker_embedding_device=(speaker_embedding_device or None),
            ref_audio_output_dir=str(Path(settings.MEDIA_ROOT) / "dubbed_videos"),
        )

        original_name = Path(video.original_video.name).name
        source_suffix = Path(original_name).suffix or ".mp4"

        with tempfile.TemporaryDirectory(prefix=f"dubbing_{video.id}_") as temp_dir:
            input_video_path = str(Path(temp_dir) / f"input{source_suffix}")
            extracted_audio_path = str(Path(temp_dir) / "extracted.wav")
            tts_audio_path = str(Path(temp_dir) / "tts.wav")
            output_video_path = str(Path(temp_dir) / f"output{source_suffix}")

            with video.original_video.open("rb") as source_file, open(input_video_path, "wb") as dst:
                shutil.copyfileobj(source_file, dst)

            result = pipeline.run(
                input_video_path=input_video_path,
                extracted_audio_path=extracted_audio_path,
                tts_audio_path=tts_audio_path,
                output_video_path=output_video_path,
            )

            logger.info("Whisper transcript: %s", result.get("transcript_text"))
            logger.info("Translated text: %s", result.get("translated_text"))


            output_name = f"dubbed_{video.id}_{Path(original_name).stem}{source_suffix}"
            with open(output_video_path, "rb") as output_file:
                video.dubbed_video.save(output_name, File(output_file), save=False)

        video.status = Video.STATUS_COMPLETED
        video.error_message = ""
        video.save(update_fields=["dubbed_video", "status", "error_message"])

        return {
            "video_id": video.id,
            "status": video.status,
            "detected_language": result.get("detected_language"),
            "segments_count": result.get("segments_count"),
        }
    except Exception as exc:
        video.status = Video.STATUS_FAILED
        video.error_message = str(exc)
        video.save(update_fields=["status", "error_message"])
        return {"video_id": video.id, "status": video.status, "error": str(exc)}
