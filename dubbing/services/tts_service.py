import wave


class CoquiTTSService:
    def __init__(self, model_name):
        self.model_name = model_name
        self._tts = None
        self._mms_tokenizer = None
        self._mms_model = None

    def synthesize_to_file(self, text, output_audio_path):
        normalized = (text or "").strip()
        if not normalized:
            raise RuntimeError("No text provided for TTS synthesis")

        if self._is_mms_model():
            self._synthesize_with_mms_vits(normalized, output_audio_path)
            return output_audio_path

        tts = self._get_coqui_tts()
        tts.tts_to_file(text=normalized, file_path=output_audio_path)
        return output_audio_path

    def _is_mms_model(self):
        return (self.model_name or "").startswith("facebook/mms-tts-")

    def _synthesize_with_mms_vits(self, text, output_audio_path):
        tokenizer, model, torch = self._get_mms_tts()

        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            output = model(**inputs).waveform

        waveform = output.squeeze().cpu().numpy()
        waveform = waveform.clip(-1.0, 1.0)
        pcm = (waveform * 32767.0).astype("int16")
        sample_rate = int(getattr(model.config, "sampling_rate", 16000))

        with wave.open(output_audio_path, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm.tobytes())

    def _get_mms_tts(self):
        if self._mms_tokenizer is not None and self._mms_model is not None:
            import torch

            return self._mms_tokenizer, self._mms_model, torch

        try:
            import torch
            from transformers import AutoTokenizer, VitsModel
        except ImportError as exc:
            raise RuntimeError(
                "MMS VITS dependencies are missing. Install with: pip install transformers torch"
            ) from exc

        self._mms_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._mms_model = VitsModel.from_pretrained(self.model_name)
        return self._mms_tokenizer, self._mms_model, torch

    def _get_coqui_tts(self):
        if self._tts is not None:
            return self._tts

        try:
            from TTS.api import TTS
        except ImportError as exc:
            raise RuntimeError(
                "coqui TTS package is not installed. Install with: pip install TTS"
            ) from exc

        self._tts = TTS(model_name=self.model_name)
        return self._tts
