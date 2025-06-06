import whisper
import time
from pydub import AudioSegment
from core.config import WHISPER_MODEL
from core.agents.stt_corrector_agent import SttCorrectorAgent

# 👇 Load model once at module level
print("🧠 Loading Whisper model at server startup:", WHISPER_MODEL)
load_start = time.time()
whisper_model = whisper.load_model(WHISPER_MODEL)
print(f"✅ Whisper model loaded in {round(time.time() - load_start, 2)}s\n")

# Convert MP3 to WAV (if you need it) -- not used
def convert_mp3_to_wav(mp3_path: str, wav_path: str):
    audio = AudioSegment.from_mp3(mp3_path)
    audio.export(wav_path, format="wav")

# Transcription function (uses preloaded model)
def transcribe_audio(audio_path: str, correct: bool = False) -> str:
    print("🛠️ > Transcribing audio:", audio_path)

    result = whisper_model.transcribe(audio_path)
    text = result['text']
    print("🛠️ > Transcription result:", text)

    if correct:
        return SttCorrectorAgent(text)

    return text
