class SpeakerDiarizationService:
    def __init__(self, model_name="pyannote/speaker-diarization-3.1", auth_token=None):
        self.model_name = model_name
        self.auth_token = auth_token
        self._pipeline = None

    def diarize(self, audio_path, min_segment_duration=0.4, min_speakers=None, max_speakers=None):
        pipeline = self._get_pipeline()
        kwargs = {}
        if min_speakers is not None and str(min_speakers).strip() != "":
            kwargs["min_speakers"] = int(min_speakers)
        if max_speakers is not None and str(max_speakers).strip() != "":
            kwargs["max_speakers"] = int(max_speakers)
        diarization = pipeline(audio_path, **kwargs) if kwargs else pipeline(audio_path)

        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            start = float(turn.start)
            end = float(turn.end)
            if end - start < min_segment_duration:
                continue
            segments.append(
                {
                    "start": start,
                    "end": end,
                    "speaker": str(speaker),
                }
            )

        if not segments:
            duration = self._get_total_duration(diarization)
            if duration > 0:
                segments.append({"start": 0.0, "end": duration, "speaker": "SPEAKER_00"})

        return sorted(segments, key=lambda x: x["start"])

    def _get_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline

        self._ensure_torchaudio_compat()

        try:
            from pyannote.audio import Pipeline
        except ImportError as exc:
            raise RuntimeError(
                "pyannote.audio is not installed. Install with: pip install pyannote.audio"
            ) from exc

        kwargs = {}
        if self.auth_token:
            kwargs["use_auth_token"] = self.auth_token
        self._pipeline = Pipeline.from_pretrained(self.model_name, **kwargs)
        if self._pipeline is None:
            raise RuntimeError(
                "Failed to load diarization pipeline. Accept model terms on "
                "https://huggingface.co/pyannote/speaker-diarization-3.1 "
                "and set HUGGINGFACE_TOKEN in .env."
            )
        return self._pipeline

    def _ensure_torchaudio_compat(self):
        try:
            import torchaudio
        except Exception:
            return

        if not hasattr(torchaudio, "set_audio_backend"):
            torchaudio.set_audio_backend = lambda *args, **kwargs: None

    def _get_total_duration(self, diarization):
        end = 0.0
        for segment in diarization.get_timeline():
            end = max(end, float(segment.end))
        return end
