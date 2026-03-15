import logging
from pathlib import Path
import os


logger = logging.getLogger(__name__)


class SpeakerEmbeddingService:
    def __init__(
        self,
        model_name="pyannote/embedding",
        cache_dir=None,
        auth_token=None,
        device=None,
        local_files_only=True,
    ):
        self.model_name = model_name
        self.cache_dir = (str(cache_dir).strip() if cache_dir else "") or None
        self.auth_token = auth_token
        self.device = device
        self.local_files_only = bool(local_files_only)
        self._inference = None

        # Prefer offline by default; if the model isn't in cache, fail with a clear error.
        os.environ.setdefault("HF_HUB_OFFLINE", "1" if self.local_files_only else "0")
        if self.cache_dir:
            # huggingface_hub expects cache_dir to point at the "hub" cache directory.
            os.environ.setdefault("HF_HUB_CACHE", self.cache_dir)
            os.environ.setdefault("HUGGINGFACE_HUB_CACHE", self.cache_dir)

    def embed(self, wav_path):
        import numpy as np

        inference = self._get_inference()
        emb = inference(str(wav_path))
        # Normalize output to a 1D numpy float32 vector on CPU to avoid torch/numpy mixing issues.
        if hasattr(emb, "detach"):
            emb = emb.detach().cpu().numpy()
        try:
            emb = emb.squeeze()
        except Exception:
            pass
        emb = np.asarray(emb, dtype="float32").reshape(-1)
        return emb

    def _get_inference(self):
        if self._inference is not None:
            return self._inference

        self._ensure_torchaudio_compat()

        try:
            from pyannote.audio import Inference, Model
        except ImportError as exc:
            raise RuntimeError(
                "pyannote.audio is required for speaker embeddings. Install with: pip install pyannote.audio"
            ) from exc

        kwargs = {}
        if self.auth_token:
            kwargs["use_auth_token"] = self.auth_token

        # NOTE: pyannote.audio Model.from_pretrained expects a HF repo id, not a local folder
        # produced via `huggingface-cli download --local-dir ...`.
        model_ref = self.model_name
        try:
            if self.local_files_only:
                kwargs["local_files_only"] = True
            if self.cache_dir:
                kwargs["cache_dir"] = self.cache_dir

            model = Model.from_pretrained(model_ref, **kwargs)
        except Exception as exc:
            hint = (
                "Speaker embedding model is not available locally.\n"
                "Fix options:\n"
                "1) Download it into the default HF cache (recommended), then rerun.\n"
                "   IMPORTANT: do NOT use --local-dir for pyannote; it must be in the HF cache structure.\n"
                "2) Or set SPEAKER_EMBEDDING_CACHE_DIR to your HF hub cache directory and rerun.\n"
                "\n"
                "Example:\n"
                "  huggingface-cli login\n"
                "  huggingface-cli download pyannote/embedding\n"
                "  # Optional if your cache is non-default:\n"
                "  # SPEAKER_EMBEDDING_CACHE_DIR=C:\\Users\\AKBAR\\.cache\\huggingface\\hub\n"
                "\n"
                f"Details: model_ref={model_ref!r} error={exc!s}\n"
            )
            raise RuntimeError(hint) from exc
        self._inference = Inference(model, window="whole", device=self.device)
        return self._inference

    def _ensure_torchaudio_compat(self):
        try:
            import torchaudio
        except Exception:
            return

        if not hasattr(torchaudio, "set_audio_backend"):
            torchaudio.set_audio_backend = lambda *args, **kwargs: None


def cosine_similarity(a, b):
    import numpy as np

    if hasattr(a, "detach"):
        a = a.detach().cpu().numpy()
    if hasattr(b, "detach"):
        b = b.detach().cpu().numpy()

    a = np.asarray(a, dtype="float32").reshape(-1)
    b = np.asarray(b, dtype="float32").reshape(-1)
    an = np.linalg.norm(a) + 1e-8
    bn = np.linalg.norm(b) + 1e-8
    return float((a @ b) / (an * bn))
