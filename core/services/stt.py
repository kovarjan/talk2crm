import whisper
from pydub import AudioSegment
from core.config import WHISPER_MODEL
from core.agents.stt_corrector_agent import SttCorrectorAgent

# Convert MP3 to WAV
def convert_mp3_to_wav(mp3_path: str, wav_path: str):
    audio = AudioSegment.from_mp3(mp3_path)
    audio.export(wav_path, format="wav")


def transcribe_audio(audio_path: str, correct: bool = False) -> str:
    print("🛠️ > Loading Whisper model:", WHISPER_MODEL)
    model = whisper.load_model(WHISPER_MODEL)

    # if is mp3 convert to wav
    if audio_path.endswith('.mp3'):
        wav_path = "temp.wav"
        convert_mp3_to_wav(audio_path, wav_path)
        audio_path = wav_path
        print("🛠️ > Converted MP3 to WAV")

    print("🛠️ > Transcribing audio:", audio_path)
    result = model.transcribe(audio_path)
    
    result = result['text']

    print("🛠️ > Transcription src:", result)

    if correct:
        return SttCorrectorAgent(result)

    return result