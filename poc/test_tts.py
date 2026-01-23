from transformers import AutoProcessor, SeamlessM4Tv2Model
import scipy.io.wavfile
import torch

# Check if CUDA (GPU) is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device}")  # Should say 'cuda'

processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")

# Move the massive 9GB model to the GPU
model = model.to(device)

# Process text
text_inputs = processor(text="Dobrý den, s čím vám můžu pomoci? TTS.", src_lang="ces", return_tensors="pt")

# Move inputs to the GPU as well
text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

# Generate audio (this happens on GPU now)
audio_array_from_text = model.generate(**text_inputs, tgt_lang="ces")[0].cpu().numpy().squeeze()

sample_rate = model.config.sampling_rate
scipy.io.wavfile.write("out_from_text_gpu.wav", rate=sample_rate, data=audio_array_from_text)
print("Done! Saved out_from_text_gpu.wav")