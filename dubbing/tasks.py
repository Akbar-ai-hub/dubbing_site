import os
import shutil
import tempfile
from pathlib import Path

from celery import shared_task
from django.conf import settings
from django.core.files.base import File

from videos.models import Video

from .services import DubbingPipelineService


def _get_setting(name, default=None):
    return getattr(settings, name, os.environ.get(name, default))


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
        whisper_model = _get_setting("WHISPER_MODEL_NAME", "base")
        translation_model = _get_setting(
            "HF_TRANSLATION_MODEL_NAME", "Helsinki-NLP/opus-mt-en-ru"
        )
        tts_model = _get_setting("COQUI_TTS_MODEL_NAME", "tts_models/en/ljspeech/tacotron2-DDC")
        source_language = _get_setting("DUBBING_SOURCE_LANGUAGE", None)

        pipeline = DubbingPipelineService(
            ffmpeg_bin=ffmpeg_bin,
            whisper_model_name=whisper_model,
            translation_model_name=translation_model,
            tts_model_name=tts_model,
            source_language=source_language,
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
        }
    except Exception as exc:
        video.status = Video.STATUS_FAILED
        video.error_message = str(exc)
        video.save(update_fields=["status", "error_message"])
        return {"video_id": video.id, "status": video.status, "error": str(exc)}
