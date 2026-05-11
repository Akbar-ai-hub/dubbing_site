import requests


class WhisperService:
    def __init__(
        self,
        model_name="whisper-large-v3",
        api_key="",
        base_url="https://api.groq.com/openai/v1/audio/transcriptions",
        timeout_sec=180,
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.timeout_sec = int(timeout_sec)

    def transcribe(self, audio_path, language=None):
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY is not configured")

        data = {"model": self.model_name, "response_format": "verbose_json"}
        data["timestamp_granularities[]"] = ["segment", "word"]
        if language:
            data["language"] = language

        headers = {"Authorization": f"Bearer {self.api_key}"}
        with open(audio_path, "rb") as audio_file:
            files = {"file": (audio_path, audio_file, "audio/wav")}
            response = requests.post(
                self.base_url,
                headers=headers,
                data=data,
                files=files,
                timeout=self.timeout_sec,
            )

        if response.status_code >= 400:
            raise RuntimeError(
                f"Groq ASR request failed: {response.status_code} {response.text}"
            )

        payload = response.json()
        return {
            "text": (payload.get("text") or "").strip(),
            "language": (payload.get("language") or language or "").strip() or language,
            "segments": self._normalize_segments(payload.get("segments")),
            "words": self._normalize_words(payload.get("words")),
        }

    def _normalize_segments(self, raw_segments):
        if not isinstance(raw_segments, list):
            return []

        normalized = []
        for seg in raw_segments:
            if not isinstance(seg, dict):
                continue
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            try:
                start = float(seg.get("start", 0.0))
                end = float(seg.get("end", 0.0))
            except (TypeError, ValueError):
                continue
            if end <= start:
                continue
            normalized.append({"start": start, "end": end, "text": text})

        return sorted(normalized, key=lambda item: item["start"])

    def _normalize_words(self, raw_words):
        if not isinstance(raw_words, list):
            return []

        normalized = []
        for word in raw_words:
            if not isinstance(word, dict):
                continue
            text = (word.get("word") or word.get("text") or "").strip()
            if not text:
                continue
            try:
                start = float(word.get("start", 0.0))
                end = float(word.get("end", 0.0))
            except (TypeError, ValueError):
                continue
            if end <= start:
                continue
            normalized.append({"word": text, "start": start, "end": end})

        return sorted(normalized, key=lambda item: item["start"])
