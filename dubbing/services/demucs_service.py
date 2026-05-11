import shutil
import subprocess
import sys
from pathlib import Path


class DemucsService:
    def __init__(self, python_bin=None, model_name="htdemucs"):
        # Use the current interpreter by default to match the venv used by the app.
        if not python_bin:
            python_bin = sys.executable or "python"
        self.python_bin = python_bin
        self.model_name = model_name

    def is_available(self):
        return shutil.which(self.python_bin) is not None

    def _separate(self, input_audio_path, output_dir):
        out_dir = Path(output_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        command = [
            self.python_bin,
            "-m",
            "demucs.separate",
            "-n",
            self.model_name,
            "--two-stems",
            "vocals",
            "-o",
            str(out_dir),
            str(input_audio_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            raise RuntimeError(f"Demucs failed. {stderr}")

        in_path = Path(input_audio_path)
        base = out_dir / self.model_name / in_path.stem
        return base

    def separate_vocals(self, input_audio_path, output_dir):
        """
        Runs demucs to extract vocals stem.
        Returns path to vocals wav if successful.
        """
        base = self._separate(input_audio_path, output_dir)
        vocals_path = base / "vocals.wav"
        if not vocals_path.exists():
            raise RuntimeError(f"Demucs vocals output not found: {vocals_path}")
        return str(vocals_path)

    def separate_vocals_and_background(self, input_audio_path, output_dir):
        """
        Runs demucs once and returns both vocals and background stems.
        """
        base = self._separate(input_audio_path, output_dir)
        vocals_path = base / "vocals.wav"
        bg_path = base / "no_vocals.wav"
        if not vocals_path.exists():
            raise RuntimeError(f"Demucs vocals output not found: {vocals_path}")
        if not bg_path.exists():
            raise RuntimeError(f"Demucs background output not found: {bg_path}")
        return str(vocals_path), str(bg_path)

    def separate_background(self, input_audio_path, output_dir):
        """
        Runs demucs to extract non-vocals (background) stem.
        Returns path to no_vocals wav if successful.
        """
        base = self._separate(input_audio_path, output_dir)
        bg_path = base / "no_vocals.wav"
        if not bg_path.exists():
            raise RuntimeError(f"Demucs background output not found: {bg_path}")
        return str(bg_path)
