import torch
from TTS.api import TTS

# 1. Load the model to GPU (RTX 3090)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading XTTS v2 on {device}...")

# This will download the model automatically on the first run
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

text = "Dobrý den, s čím vám můžu pomoci? Omlouvám se, požadavek není úplný. Můžete ho prosím upřesnit? Schůzka byla připravena. "

# 2. Generate Audio
# IMPORTANT: XTTS needs a reference voice to clone. 
# Since you don't have one yet, we will use one of the default speakers provided by the model.
# You can see available speakers with: print(tts.speakers)

# self.predefined_speakers = ['Claribel Dervla', 'Daisy Studious', 'Gracie Wise', 'Tammie Ema', 'Alison Dietlinde', 'Ana Florence', 'Annmarie Nele', 'Asya Anara', 'Brenda Stern', 'Gitta Nikolina', 'Henriette Usha', 'Sofia Hellen', 'Tammy Grit', 'Tanja Adelina', 'Vjollca Johnnie', 'Andrew Chipper', 'Badr Odhiambo', 'Dionisio Schuyler', 'Royston Min', 'Viktor Eka', 'Abrahan Mack', 'Adde Michal', 'Baldur Sanjin', 'Craig Gutsy', 'Damien Black', 'Gilberto Mathias', 'Ilkin Urbano', 'Kazuhiko Atallah', 'Ludvig Milivoj', 'Suad Qasim', 'Torcull Diarmuid', 'Viktor Menelaos', 'Zacharie Aimilios', 'Nova Hogarth', 'Maja Ruoho', 'Uta Obando', 'Lidiya Szekeres', 'Chandra MacFarland', 'Szofi Granger', 'Camilla Holmström', 'Lilya Stainthorpe', 'Zofija Kendrick', 'Narelle Moon', 'Barbora MacLean', 'Alexandra Hisakawa', 'Alma María', 'Rosemary Okafor', 'Ige Behringer', 'Filip Traverse', 'Damjan Chapman', 'Wulf Carlevaro', 'Aaron Dreschner', 'Kumar Dahl', 'Eugenio Mataracı', 'Ferran Simen', 'Xavier Hayasaka', 'Luis Moray', 'Marcos Rudaski']

tts.tts_to_file(
    text=text,
    speaker="Annmarie Nele", # "Ana Florence" is a good default English voice, but works for Czech too
    language="cs", 
    file_path="out_xtts_czech.wav"
)

print("Saved out_xtts_czech.wav")
