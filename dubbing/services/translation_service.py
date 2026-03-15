import os
from pathlib import Path


class LocalNLLBTranslationService:
    def __init__(
        self,
        model_dir,
        source_lang="eng_Latn",
        target_lang="kaz_Cyrl",
        batch_size=8,
        max_new_tokens=256,
        device=None,
    ):
        self.model_dir = str(model_dir)
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.batch_size = int(batch_size)
        self.max_new_tokens = int(max_new_tokens)
        self.device = device

        self._tokenizer = None
        self._model = None

        # Hard-disable network usage for HF/Transformers.
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    def translate(self, text):
        normalized = (text or "").strip()
        if not normalized:
            return ""
        return self.translate_batch([normalized])[0]

    def translate_batch(self, texts):
        items = [(t or "").strip() for t in (texts or [])]
        if not items:
            return []

        tokenizer, model, torch = self._get_model()
        tokenizer.src_lang = self.source_lang

        forced_bos_token_id = tokenizer.lang_code_to_id.get(self.target_lang)
        if forced_bos_token_id is None:
            raise RuntimeError(f"Unknown NLLB target_lang: {self.target_lang}")

        outputs = []
        for start in range(0, len(items), max(1, self.batch_size)):
            batch = items[start : start + self.batch_size]
            encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
            encoded = {k: v.to(model.device) for k, v in encoded.items()}

            with torch.no_grad():
                generated = model.generate(
                    **encoded,
                    forced_bos_token_id=forced_bos_token_id,
                    max_new_tokens=self.max_new_tokens,
                )
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
            outputs.extend([(d or "").strip() for d in decoded])

        # Fallback: preserve length and avoid empty strings.
        if len(outputs) < len(items):
            outputs.extend([""] * (len(items) - len(outputs)))
        outputs = outputs[: len(items)]
        return [out or src for out, src in zip(outputs, items)]

    def _get_model(self):
        if self._tokenizer is not None and self._model is not None:
            import torch

            return self._tokenizer, self._model, torch

        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "NLLB translation dependencies missing. Install: pip install transformers torch sentencepiece"
            ) from exc

        snapshot_dir = self._resolve_hf_snapshot_dir(self.model_dir)
        self._tokenizer = AutoTokenizer.from_pretrained(snapshot_dir, local_files_only=True)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(snapshot_dir, local_files_only=True)

        device = self.device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(device)
        self._model.eval()

        return self._tokenizer, self._model, torch

    def _resolve_hf_snapshot_dir(self, model_dir):
        path = Path(model_dir).expanduser().resolve()
        if not path.exists():
            raise RuntimeError(f"NLLB model dir not found: {path}")

        # If user points to cache root (models--.../), select latest snapshot.
        snapshots = path / "snapshots"
        if snapshots.exists() and snapshots.is_dir():
            snapshot_dirs = [p for p in snapshots.iterdir() if p.is_dir()]
            snapshot_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            if not snapshot_dirs:
                raise RuntimeError(f"No snapshots found in: {snapshots}")
            return str(snapshot_dirs[0])

        return str(path)

