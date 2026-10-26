"""Safe handling of uploaded audio: size cap, suffix allowlist, bounded concurrency."""
from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from fastapi import HTTPException, UploadFile

from app.core.config import get_settings

_CHUNK_BYTES = 1024 * 1024

# Container formats ffmpeg/faster-whisper can decode, keyed by MIME type.
CONTENT_TYPE_SUFFIXES: dict[str, str] = {
    "audio/webm": ".webm",
    "video/webm": ".webm",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/wave": ".wav",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/mp4": ".m4a",
    "audio/x-m4a": ".m4a",
    "audio/m4a": ".m4a",
    "audio/aac": ".aac",
    "audio/ogg": ".ogg",
    "audio/opus": ".ogg",
    "audio/flac": ".flac",
    "audio/x-flac": ".flac",
}
ALLOWED_SUFFIXES = frozenset(CONTENT_TYPE_SUFFIXES.values())


def resolve_audio_suffix(content_type: str | None, filename: str | None) -> str:
    """Suffix for the temp file: derived from the content type, else from the
    client filename only if it is on the allowlist. The raw client filename is
    never used. Raises 415 for anything else."""
    mime = (content_type or "").split(";", 1)[0].strip().lower()
    if mime in CONTENT_TYPE_SUFFIXES:
        return CONTENT_TYPE_SUFFIXES[mime]
    candidate = Path(filename or "").suffix.lower()
    if candidate in ALLOWED_SUFFIXES:
        return candidate
    raise HTTPException(status_code=415, detail="Unsupported audio format")


async def save_upload_to_tempfile(file: UploadFile) -> str:
    """Copies the upload to a temp file in chunks, enforcing the size cap.

    Returns the temp path; the caller owns (and must unlink) it."""
    max_bytes = get_settings().audio_max_upload_bytes
    suffix = resolve_audio_suffix(file.content_type, file.filename)
    size = 0
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        path = tmp.name
        try:
            while True:
                chunk = await file.read(_CHUNK_BYTES)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(status_code=413, detail="Audio file is too large")
                tmp.write(chunk)
        except BaseException:
            Path(path).unlink(missing_ok=True)
            raise
    if size == 0:
        Path(path).unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Empty audio file")
    return path


_semaphore: tuple[asyncio.AbstractEventLoop, asyncio.Semaphore] | None = None


def get_transcription_semaphore() -> asyncio.Semaphore:
    """Limits concurrent (CPU/GPU-bound) transcriptions per worker process."""
    global _semaphore
    loop = asyncio.get_running_loop()
    if _semaphore is None or _semaphore[0] is not loop:
        limit = max(1, get_settings().audio_max_concurrent_transcriptions)
        _semaphore = (loop, asyncio.Semaphore(limit))
    return _semaphore[1]


class BodySizeLimitMiddleware:
    """ASGI middleware: rejects oversized request bodies with 413 before the
    multipart parser or the HMAC dependency buffers them."""

    def __init__(self, app, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        started = False

        async def send_wrapper(message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        async def reject() -> None:
            body = b'{"detail":"Request body too large"}'
            await send({
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            })
            await send({"type": "http.response.body", "body": body})

        for name, value in scope.get("headers", []):
            if name == b"content-length":
                try:
                    declared = int(value)
                except ValueError:
                    declared = 0
                if declared > self.max_bytes:
                    await reject()
                    return

        received = 0

        async def receive_wrapper():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise _BodyTooLarge()
            return message

        try:
            await self.app(scope, receive_wrapper, send_wrapper)
        except _BodyTooLarge:
            if not started:
                await reject()


class _BodyTooLarge(Exception):
    pass
