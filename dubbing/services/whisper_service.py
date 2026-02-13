class WhisperService:
    def __init__(self, model_name="base"):
        self.model_name = model_name
        self._model = None

    def transcribe(self, audio_path, language=None):
        model = self._get_model()
        kwargs = {}
        if language:
            kwargs["language"] = language

        result = model.transcribe(audio_path, **kwargs)
        text = (result.get("text") or "").strip()
        detected_language = result.get("language")
        return {
            "text": text,
            "language": detected_language,
        }

    def _get_model(self):
        if self._model is not None:
            return self._model

        try:
            import whisper
        except ImportError as exc:
            raise RuntimeError(
                "whisper package is not installed. Install with: pip install openai-whisper"
            ) from exc

        self._model = whisper.load_model(self.model_name)
        return self._model
