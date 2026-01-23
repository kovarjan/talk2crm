import torch
from TTS.api import TTS
import os
import uuid
import re
from core.config import TTS_MODEL, TTS_METHOD, TTS_DEVICE, CACHE_DIR

def _resolve_device() -> str:
    choice = (TTS_DEVICE or "auto").lower().strip()
    if choice in ("cuda", "gpu"):
        return "cuda"
    if choice == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"

class TTSService:
    _instance = None
    _seamless_model = None
    _seamless_processor = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        if not TTS_MODEL:
            raise ValueError("TTS_MODEL environment variable not set.")
        self.tts = None


    def generate_speech(
        self,
        text: str,
        speaker: str = "Sofia Hellen",
        language: str = "cs",
        method: str = TTS_METHOD,
    ) -> str:
        """
        Generates speech from text and saves it to a temporary file.
        Returns the id of the audio file.
        """
        
        audio_dir = os.path.join(CACHE_DIR, "audio")
        os.makedirs(audio_dir, exist_ok=True)
        file_id = str(uuid.uuid4())
        output_path = os.path.join(audio_dir, f"{file_id}.wav")

        method = (method or TTS_METHOD or "xtts").lower().strip()
        if method == "seamless":
            self._generate_with_seamless(text, output_path)
        else:
            if self.tts is None:
                device = _resolve_device()
                print(f"Loading TTS model: {TTS_MODEL} on {device}...")
                self.tts = TTS(TTS_MODEL, progress_bar=True, gpu=(device == "cuda"))
            tts_text = _sanitize_for_tts(text, language)
            try:
                self.tts.tts_to_file(
                    text=tts_text,
                    speaker=speaker,
                    language=language,
                    file_path=output_path
                )
            except NotImplementedError:
                # Work around num2words cs ordinal expansion crashes.
                tts_text = _sanitize_for_tts(tts_text, language, drop_ordinals=True)
                self.tts.tts_to_file(
                    text=tts_text,
                    speaker=speaker,
                    language=language,
                    file_path=output_path
                )
        return file_id

    def _generate_with_seamless(self, text: str, output_path: str) -> None:
        from transformers import AutoProcessor, SeamlessM4Tv2Model
        import scipy.io.wavfile

        if self._seamless_processor is None or self._seamless_model is None:
            self._seamless_processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
            self._seamless_model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")
            target_device = _resolve_device()
            try:
                self._seamless_model = self._seamless_model.to(target_device)
            except torch.OutOfMemoryError:
                print("⚠️  SeamlessM4T OOM on GPU, falling back to CPU.")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self._seamless_model = self._seamless_model.to("cpu")

        model_device = next(self._seamless_model.parameters()).device
        text_inputs = self._seamless_processor(text=text, src_lang="ces", return_tensors="pt")
        text_inputs = {k: v.to(model_device) for k, v in text_inputs.items()}
        audio_array = self._seamless_model.generate(**text_inputs, tgt_lang="ces")[0].cpu().numpy().squeeze()
        sample_rate = self._seamless_model.config.sampling_rate
        scipy.io.wavfile.write(output_path, rate=sample_rate, data=audio_array)

def generate_speech(
    text: str,
    speaker: str = "Sofia Hellen",
    language: str = "cs",
    method: str = TTS_METHOD,
) -> str:
    """
    Function to generate speech from text.
    """
    tts_service = TTSService.get_instance()
    return tts_service.generate_speech(text, speaker, language, method=method)

def _sanitize_for_tts(text: str, language: str, drop_ordinals: bool = False) -> str:
    """
    Avoid ordinal expansions that crash num2words in Czech.
    Example: "30. 1. 2026" -> "30 1 2026".
    """
    if not text:
        return text
    if language.lower().startswith("cs"):
        s = text
        if drop_ordinals:
            s = re.sub(r"\b(\d+)\.(\s|$)", r"\1\2", s)
        else:
            s = re.sub(r"\b(\d+)\.(\s|$)", r"\1\2", s)
        return s
    return text
