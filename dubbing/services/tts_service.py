from pathlib import Path
import logging
import os


logger = logging.getLogger(__name__)


class CoquiTTSService:
    def __init__(
        self,
        xtts_local_dir,
        xtts_fallback_language="tr",
    ):
        self.xtts_local_dir = (xtts_local_dir or "").strip() or None
        if not self.xtts_local_dir:
            raise RuntimeError("COQUI_XTTS_LOCAL_DIR is required for XTTS fine-tuned model")
        self.xtts_fallback_language = (xtts_fallback_language or "tr").strip().lower()
        self._tts = None

    def synthesize_to_file(self, text, output_audio_path, speaker_reference_audio=None, language=None):
        normalized = (text or "").strip()
        if not normalized:
            raise RuntimeError("No text provided for TTS synthesis")

        if not speaker_reference_audio:
            raise RuntimeError(
                "XTTS requires a reference wav (speaker_wav). Provide speaker_reference_audio/--speaker-wav."
            )

        tts = self._get_coqui_tts()

        kwargs = {"text": normalized, "file_path": output_audio_path}
        kwargs["speaker_wav"] = speaker_reference_audio
        requested_language = (language or "").strip().lower()
        fallback_language = (self.xtts_fallback_language or "").strip().lower()
        resolved_language = requested_language or fallback_language
        if resolved_language:
            kwargs["language"] = resolved_language

        try:
            tts.tts_to_file(**kwargs)
        except Exception as exc:
            # If the fine-tuned model actually supports `kk`, we want to pass it through.
            # But if it doesn't, retry once with fallback language instead of failing the whole segment.
            message = str(exc).lower()
            if (
                requested_language
                and fallback_language
                and requested_language != fallback_language
                and ("language" in message and "not supported" in message)
            ):
                # First try to enable the language in the loaded model object (some fine-tunes add
                # language embeddings but upstream validation list may be stale/hardcoded).
                if self._try_enable_language(tts, requested_language):
                    logger.warning(
                        "XTTS language '%s' was rejected; enabled it on the loaded model and retrying.",
                        requested_language,
                    )
                    kwargs["language"] = requested_language
                    tts.tts_to_file(**kwargs)
                else:
                    logger.warning(
                        "XTTS language '%s' not supported by the loaded model; retrying with fallback '%s'.",
                        requested_language,
                        fallback_language,
                    )
                    kwargs["language"] = fallback_language
                    tts.tts_to_file(**kwargs)
            else:
                raise
        return output_audio_path

    def _try_enable_language(self, tts, language):
        lang = (language or "").strip().lower()
        if not lang:
            return False

        candidates = []
        # Common attribute layouts across Coqui TTS versions.
        candidates.append(getattr(tts, "tts_model", None))
        synthesizer = getattr(tts, "synthesizer", None)
        if synthesizer is not None:
            candidates.append(getattr(synthesizer, "tts_model", None))

        enabled = False
        for model in candidates:
            if model is None:
                continue
            langs = getattr(model, "languages", None)
            if isinstance(langs, (list, tuple)):
                if lang not in langs:
                    try:
                        langs.append(lang)
                    except Exception:
                        continue
                enabled = True

        return enabled

    def _get_coqui_tts(self):
        if self._tts is not None:
            return self._tts

        # Coqui TTS reads a user-data dir from Windows registry unless TTS_HOME is set.
        # In some environments the registry lookup can fail, so default TTS_HOME to a workspace path.
        if "TTS_HOME" not in os.environ:
            repo_root = Path(__file__).resolve().parents[3]
            tts_home = repo_root / "_tmp" / "tts_home"
            tts_home.mkdir(parents=True, exist_ok=True)
            os.environ["TTS_HOME"] = str(tts_home)

        try:
            from TTS.api import TTS
        except ImportError as exc:
            raise RuntimeError(
                "coqui TTS package is not installed. Install with: pip install TTS"
            ) from exc

        model_dir, config_path = self._resolve_xtts_paths_from_local_dir(self.xtts_local_dir)
        self._tts = TTS(model_path=model_dir, config_path=config_path)
        return self._tts

    def _resolve_xtts_paths_from_local_dir(self, local_dir):
        model_dir = Path(local_dir).expanduser().resolve()
        if not model_dir.exists():
            raise RuntimeError(f"XTTS local dir not found: {model_dir}")

        config_path = model_dir / "config.json"
        if not config_path.exists():
            raise RuntimeError(f"XTTS config.json not found in: {model_dir}")

        self._ensure_model_pth_exists(model_dir)
        return str(model_dir), str(config_path)

    def _ensure_model_pth_exists(self, model_dir):
        model_dir = Path(model_dir)
        model_pth = model_dir / "model.pth"
        if model_pth.exists():
            return

        best_model = model_dir / "best_model.pth"
        if best_model.exists():
            raise RuntimeError(
                f"XTTS expects 'model.pth' inside {model_dir}, but only 'best_model.pth' was found. "
                "Rename/copy best_model.pth -> model.pth."
            )

        raise RuntimeError(f"XTTS model.pth not found in {model_dir}")
