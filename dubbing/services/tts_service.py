from pathlib import Path
import logging

import numpy as np
import soundfile as sf
import torch


logger = logging.getLogger(__name__)


class CoquiTTSService:
    def __init__(
        self,
        xtts_local_dir,
        temperature=0.5,
        length_penalty=1.0,
        repetition_penalty=3.0,
        top_k=50,
        top_p=0.9,
        gpt_cond_len=6,
        max_ref_length=10,
    ):
        self.xtts_local_dir = (xtts_local_dir or "").strip() or None
        if not self.xtts_local_dir:
            raise RuntimeError("COQUI_XTTS_LOCAL_DIR is required for XTTS fine-tuned model")
        self.temperature = float(temperature)
        self.length_penalty = float(length_penalty)
        self.repetition_penalty = float(repetition_penalty)
        self.top_k = int(top_k)
        self.top_p = float(top_p)
        self.gpt_cond_len = max(1, int(gpt_cond_len))
        self.max_ref_length = max(1, int(max_ref_length))
        self._model = None
        self._config = None
        self._conditioning_cache = {}

    def synthesize_to_file(self, text, output_audio_path, speaker_reference_audio=None, language=None):
        normalized = " ".join((text or "").split()).strip()
        if not normalized:
            raise RuntimeError("No text provided for TTS synthesis")
        if not speaker_reference_audio:
            raise RuntimeError(
                "XTTS requires a reference wav (speaker_wav). Provide speaker_reference_audio/--speaker-wav."
            )

        model, config = self._get_xtts_model()
        requested_language = (language or "").strip().lower() or "kk"
        self._ensure_language(config, requested_language)

        gpt_cond_latent, speaker_embedding = self._get_conditioning_latents(
            model=model,
            speaker_reference_audio=speaker_reference_audio,
        )

        logger.info(
            "XTTS synth: language=%s speaker_wav=%s text=%s",
            requested_language,
            speaker_reference_audio,
            normalized,
        )
        with torch.no_grad():
            output = model.inference(
                text=normalized,
                language=requested_language,
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
                temperature=self.temperature,
                length_penalty=self.length_penalty,
                repetition_penalty=self.repetition_penalty,
                top_k=self.top_k,
                top_p=self.top_p,
            )

        final_wav = np.asarray(output["wav"], dtype=np.float32)
        if final_wav.size == 0:
            raise RuntimeError("XTTS produced empty audio")
        output_path = Path(output_audio_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), final_wav, 24000)
        return str(output_path)

    def _get_xtts_model(self):
        if self._model is not None and self._config is not None:
            return self._model, self._config

        try:
            from TTS.tts.configs.xtts_config import XttsConfig
            from TTS.tts.models.xtts import Xtts
        except ImportError as exc:
            raise RuntimeError(
                "coqui XTTS dependencies are not installed. Install with: pip install TTS soundfile"
            ) from exc

        model_dir = Path(self.xtts_local_dir).expanduser().resolve()
        if not model_dir.exists():
            raise RuntimeError(f"XTTS local dir not found: {model_dir}")

        config_path = model_dir / "config.json"
        vocab_path = model_dir / "vocab.json"
        model_path = model_dir / "model.pth"
        if not config_path.exists():
            raise RuntimeError(f"XTTS config.json not found in: {model_dir}")
        if not vocab_path.exists():
            raise RuntimeError(f"XTTS vocab.json not found in: {model_dir}")
        if not model_path.exists():
            raise RuntimeError(f"XTTS model.pth not found in: {model_dir}")

        config = XttsConfig()
        config.load_json(str(config_path))
        model = Xtts.init_from_config(config)
        model.load_checkpoint(
            config,
            checkpoint_dir=str(model_dir),
            checkpoint_path=str(model_path),
            vocab_path=str(vocab_path),
            eval=True,
            strict=False,
        )
        if torch.cuda.is_available():
            model.cuda()
            logger.info("XTTS model loaded on GPU")
        else:
            logger.info("XTTS model loaded on CPU")

        self._model = model
        self._config = config
        return self._model, self._config

    def _ensure_language(self, config, language):
        languages = getattr(config, "languages", None)
        if isinstance(languages, list) and language not in languages:
            languages.append(language)

    def _get_conditioning_latents(self, model, speaker_reference_audio):
        cache_key = str(Path(speaker_reference_audio).expanduser().resolve())
        cached = self._conditioning_cache.get(cache_key)
        if cached is not None:
            return cached

        latents = model.get_conditioning_latents(
            audio_path=[speaker_reference_audio],
            gpt_cond_len=self.gpt_cond_len,
            max_ref_length=self.max_ref_length,
        )
        self._conditioning_cache[cache_key] = latents
        return latents
