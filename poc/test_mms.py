from transformers import VitsModel, AutoTokenizer
import torch
import scipy.io.wavfile

# Load the Czech-specific version of MMS
model = VitsModel.from_pretrained("facebook/mms-tts-ces")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-ces")

text = "Dobrý den, s čím vám můžu pomoci? MMS."
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    output = model(**inputs).waveform

# Save to file
scipy.io.wavfile.write("out_mms_czech.wav", rate=model.config.sampling_rate, data=output.numpy().T)
