import logging
import gc
from pathlib import Path

import requests


logger = logging.getLogger(__name__)


class WhisperService:
    def __init__(
        self,
        model_name="whisper-large-v3",
        api_key="",
        base_url="https://api.groq.com/openai/v1/audio/transcriptions",
        timeout_sec=180,
        local_model_dir=None,
        local_device=None,
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.timeout_sec = int(timeout_sec)
        self.local_model_dir = str(local_model_dir or "").strip()
        self.local_device = str(local_device or "").strip()
        self._local_pipeline = None

    def transcribe(self, audio_path, language=None):
        if self.local_model_dir:
            return self._transcribe_local(audio_path, language=language)
        return self._transcribe_groq(audio_path, language=language)

    def release(self):
        if self._local_pipeline is None:
            return

        try:
            del self._local_pipeline
        except Exception:
            pass
        self._local_pipeline = None

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception as exc:
            logger.debug("Local Whisper CUDA cleanup skipped: %s", exc)

        gc.collect()
        logger.info("Local Whisper resources released")

    def _transcribe_groq(self, audio_path, language=None):
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

    def _transcribe_local(self, audio_path, language=None):
        pipe = self._get_local_pipeline()
        generate_kwargs = {"task": "transcribe"}
        local_language = self._normalize_local_language(language)
        if local_language:
            generate_kwargs["language"] = local_language

        logger.info(
            "Local Whisper transcribe: model_dir=%s language=%s audio=%s",
            self._resolve_local_model_path(),
            local_language or "",
            audio_path,
        )
        payload = pipe(
            str(audio_path),
            return_timestamps="word",
            chunk_length_s=30,
            stride_length_s=5,
            generate_kwargs=generate_kwargs,
        )

        text = (payload.get("text") or "").strip() if isinstance(payload, dict) else ""
        chunks = payload.get("chunks") if isinstance(payload, dict) else []
        words, segments = self._normalize_local_chunks(chunks)
        if not segments and words:
            segments = self._build_segments_from_words(words)

        return {
            "text": text or self._join_word_text(words),
            "language": language or "",
            "segments": segments,
            "words": words,
        }

    def _normalize_local_language(self, language):
        value = str(language or "").strip().lower()
        if not value:
            return ""
        mapping = {
            "en": "english",
            "eng": "english",
            "eng_latn": "english",
            "kk": "kazakh",
            "kaz": "kazakh",
            "kaz_cyrl": "kazakh",
        }
        return mapping.get(value, value)

    def _get_local_pipeline(self):
        if self._local_pipeline is not None:
            return self._local_pipeline

        try:
            import torch
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
        except Exception as exc:
            raise RuntimeError(
                "Local Whisper requires torch and transformers to be installed."
            ) from exc

        model_path = self._resolve_local_model_path()
        device_name = self.local_device
        if not device_name:
            device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if str(device_name).startswith("cuda") else torch.float32

        model_kwargs = {
            "torch_dtype": torch_dtype,
            "local_files_only": True,
            "attn_implementation": "eager",
        }
        try:
            model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path, **model_kwargs)
        except TypeError:
            model_kwargs.pop("attn_implementation", None)
            model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path, **model_kwargs)
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)

        pipeline_device = -1
        if str(device_name).startswith("cuda"):
            try:
                pipeline_device = int(str(device_name).split(":", 1)[1])
            except Exception:
                pipeline_device = 0

        self._local_pipeline = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=pipeline_device,
        )
        logger.info("Local Whisper model loaded from %s on %s", model_path, device_name)
        return self._local_pipeline

    def _resolve_local_model_path(self):
        root = Path(self.local_model_dir).expanduser()
        if (root / "config.json").exists():
            return str(root)

        refs_main = root / "refs" / "main"
        if refs_main.exists():
            revision = refs_main.read_text(encoding="utf-8").strip()
            snapshot = root / "snapshots" / revision
            if (snapshot / "config.json").exists():
                return str(snapshot)

        snapshots_dir = root / "snapshots"
        if snapshots_dir.exists():
            snapshots = sorted(
                [path for path in snapshots_dir.iterdir() if (path / "config.json").exists()],
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            if snapshots:
                return str(snapshots[0])

        raise RuntimeError(
            f"Local Whisper model config.json was not found under '{self.local_model_dir}'."
        )

    def _normalize_local_chunks(self, raw_chunks):
        if not isinstance(raw_chunks, list):
            return [], []

        words = []
        phrase_segments = []
        for chunk in raw_chunks:
            if not isinstance(chunk, dict):
                continue
            text = (chunk.get("text") or chunk.get("word") or "").strip()
            timestamp = chunk.get("timestamp") or chunk.get("timestamps")
            parsed = self._parse_timestamp(timestamp)
            if not text or parsed is None:
                continue
            start, end = parsed
            item = {"start": start, "end": end, "text": text}
            phrase_segments.append(item)
            words.append({"word": text, "start": start, "end": end})

        return self._normalize_words(words), self._normalize_segments(phrase_segments)

    def _parse_timestamp(self, timestamp):
        if not isinstance(timestamp, (list, tuple)) or len(timestamp) < 2:
            return None
        start, end = timestamp[0], timestamp[1]
        if start is None or end is None:
            return None
        try:
            start = float(start)
            end = float(end)
        except (TypeError, ValueError):
            return None
        if end <= start:
            return None
        return start, end

    def _build_segments_from_words(self, words):
        segments = []
        current = []
        for word in words:
            if current:
                gap = float(word["start"]) - float(current[-1]["end"])
                if gap >= 0.9:
                    segments.append(self._word_group_to_segment(current))
                    current = []
            current.append(word)
            text = str(word.get("word") or "")
            if text.endswith((".", "!", "?")):
                segments.append(self._word_group_to_segment(current))
                current = []
        if current:
            segments.append(self._word_group_to_segment(current))
        return [segment for segment in segments if segment]

    def _word_group_to_segment(self, words):
        if not words:
            return None
        return {
            "start": float(words[0]["start"]),
            "end": float(words[-1]["end"]),
            "text": self._join_word_text(words),
        }

    def _join_word_text(self, words):
        text = ""
        for item in words:
            word = str(item.get("word") or item.get("text") or "").strip()
            if not word:
                continue
            if not text or word[:1] in ".,!?;:%)]}":
                text += word
            else:
                text += " " + word
        return text.strip()

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
