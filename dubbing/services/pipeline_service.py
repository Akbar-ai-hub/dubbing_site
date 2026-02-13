from .ffmpeg_service import FFmpegService
from .translation_service import HuggingFaceTranslationService
from .tts_service import CoquiTTSService
from .whisper_service import WhisperService


class DubbingPipelineService:
    def __init__(
        self,
        ffmpeg_bin,
        whisper_model_name,
        translation_model_name,
        tts_model_name,
        source_language=None,
    ):
        self.ffmpeg_service = FFmpegService(ffmpeg_bin=ffmpeg_bin)
        self.whisper_service = WhisperService(model_name=whisper_model_name)
        self.translation_service = HuggingFaceTranslationService(
            model_name=translation_model_name
        )
        self.tts_service = CoquiTTSService(model_name=tts_model_name)
        self.source_language = source_language

    def run(self, input_video_path, extracted_audio_path, tts_audio_path, output_video_path):
        self.ffmpeg_service.extract_audio(
            input_video_path=input_video_path,
            output_audio_path=extracted_audio_path,
        )

        transcription = self.whisper_service.transcribe(
            audio_path=extracted_audio_path,
            language=self.source_language,
        )
        transcript_text = transcription["text"]
        translated_text = self.translation_service.translate(transcript_text)

        self.tts_service.synthesize_to_file(
            text=translated_text,
            output_audio_path=tts_audio_path,
        )

        self.ffmpeg_service.mux_audio_with_video(
            input_video_path=input_video_path,
            input_audio_path=tts_audio_path,
            output_video_path=output_video_path,
        )

        return {
            "transcript_text": transcript_text,
            "translated_text": translated_text,
            "detected_language": transcription.get("language"),
            "output_video_path": output_video_path,
        }
