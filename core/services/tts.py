# TODO: implement This !

from TTS.api import TTS
from talk2text.config import TTS_MODEL

tts = TTS(model_name=TTS_MODEL)

def speak_to_file(text: str, output_path="output.wav"):
    tts.tts_to_file(text=text, file_path=output_path)
