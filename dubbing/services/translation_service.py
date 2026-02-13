class HuggingFaceTranslationService:
    def __init__(self, model_name):
        self.model_name = model_name
        self._translator = None

    def translate(self, text):
        normalized = (text or "").strip()
        if not normalized:
            return ""

        if not self.model_name:
            return normalized

        translator = self._get_translator()
        translated = translator(normalized)
        if not translated:
            return normalized
        return (translated[0].get("translation_text") or "").strip() or normalized

    def _get_translator(self):
        if self._translator is not None:
            return self._translator

        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError(
                "transformers package is not installed. Install with: pip install transformers sentencepiece"
            ) from exc

        self._translator = pipeline(
            task="translation",
            model=self.model_name,
            tokenizer=self.model_name,
        )
        return self._translator
