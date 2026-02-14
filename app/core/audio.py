from __future__ import annotations

import asyncio
import os
import wave
from pathlib import Path

from app.core.config import get_settings
from app.core.logging import get_logger

try:
    from faster_whisper import WhisperModel
except Exception:  # pragma: no cover - optional dependency in local dev
    WhisperModel = None

try:
    import edge_tts
except Exception:  # pragma: no cover - optional dependency in local dev
    edge_tts = None


logger = get_logger(__name__)
_whisper_model = None


def _maybe_add_cuda_lib_paths() -> None:
    # When using pip-provided NVIDIA libs, expose them to runtime linker.
    paths: list[str] = []
    try:
        import nvidia.cublas.lib as cublas_lib  # type: ignore
        paths.append(str(Path(cublas_lib.__file__).resolve().parent))
    except Exception:
        pass
    try:
        import nvidia.cudnn.lib as cudnn_lib  # type: ignore
        paths.append(str(Path(cudnn_lib.__file__).resolve().parent))
    except Exception:
        pass
    if not paths:
        return
    current = os.environ.get("LD_LIBRARY_PATH", "")
    additions = [p for p in paths if p and p not in current]
    if additions:
        os.environ["LD_LIBRARY_PATH"] = ":".join(additions + ([current] if current else []))


def _resolve_whisper_runtime(device: str, compute_type: str) -> tuple[str, str]:
    selected_device = (device or "auto").strip().lower()
    selected_compute = (compute_type or "auto").strip().lower()

    if selected_device == "auto":
        cuda_devices = 0
        try:
            import ctranslate2  # type: ignore
            cuda_devices = int(ctranslate2.get_cuda_device_count())
        except Exception:
            cuda_devices = 0
        selected_device = "cuda" if cuda_devices > 0 else "cpu"

    if selected_compute == "auto":
        selected_compute = "float16" if selected_device == "cuda" else "int8"

    return selected_device, selected_compute


def _get_whisper_model():
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model

    settings = get_settings()
    if WhisperModel is None:
        raise RuntimeError(
            "faster_whisper is not installed. Add it to requirements and reinstall."
        )

    _maybe_add_cuda_lib_paths()
    device, compute_type = _resolve_whisper_runtime(
        settings.whisper_device, settings.whisper_compute_type
    )

    try:
        _whisper_model = WhisperModel(
            settings.whisper_model_size,
            device=device,
            compute_type=compute_type,
        )
        logger.info(
            "Whisper initialized model=%s device=%s compute_type=%s",
            settings.whisper_model_size,
            device,
            compute_type,
        )
    except Exception:
        if device == "cuda" and settings.whisper_allow_cpu_fallback:
            logger.exception(
                "Whisper CUDA init failed, falling back to CPU model=%s",
                settings.whisper_model_size,
            )
            _whisper_model = WhisperModel(
                settings.whisper_model_size,
                device="cpu",
                compute_type="int8",
            )
            logger.info(
                "Whisper initialized model=%s device=cpu compute_type=int8 (fallback)",
                settings.whisper_model_size,
            )
        else:
            raise
    return _whisper_model


def transcribe(file_path: str, language: str | None = None) -> str:
    model = _get_whisper_model()
    segments, _ = model.transcribe(file_path, language=language)
    text = " ".join(segment.text.strip() for segment in segments).strip()
    logger.info("Transcribed audio file=%s chars=%s", file_path, len(text))
    return text


def _write_fallback_silence_wav(output_path: str, seconds: float = 1.0) -> None:
    sample_rate = 16000
    frame_count = int(sample_rate * seconds)
    silence = b"\x00\x00" * frame_count
    with wave.open(output_path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(silence)


async def synthesize_to_file(text: str, output_path: str, voice: str | None = None) -> None:
    settings = get_settings()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    chosen_voice = voice or settings.tts_voice
    provider = settings.tts_provider.lower()

    if provider == "edge-tts" and edge_tts is not None:
        communicate = edge_tts.Communicate(text=text, voice=chosen_voice)
        await communicate.save(str(output))
        logger.info("Generated TTS audio with edge-tts at %s", output_path)
        return

    logger.warning(
        "TTS provider '%s' unavailable, writing fallback silence wav", provider
    )
    _write_fallback_silence_wav(str(output))


def synthesize_to_file_sync(text: str, output_path: str, voice: str | None = None) -> None:
    asyncio.run(synthesize_to_file(text=text, output_path=output_path, voice=voice))
