from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import shutil
import os
from core.pipelines.command_pipeline import process_voice_command


app = FastAPI()

# Allow requests from your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["*"] for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Replace this with your real processing function
def process_audio_and_generate_crm_call(file_path: str) -> dict:
    # check if .env file exists
    if not os.path.exists(".env"):
        print("⚠️ .env file not found. Please create a .env file with the required environment variables.")
        exit(1)
    
    # audio_path = "assets/audio/untitled_exmaple2.mp3"
    # audio_path = "data/examples/audio/activity1_normal_grade1.mp3"
    # audio_path = "data/examples/audio/activity3_normal_grade1.mp3"
    audio_path = file_path

    print("audio_path:", audio_path)
    
    # Process the voice command
    response = process_voice_command(audio_path)

    # save recoding to file data/recordings
    recordings_dir = "data/recordings"
    os.makedirs(recordings_dir, exist_ok=True)
    recording_file_path = os.path.join(recordings_dir, os.path.basename(file_path))
    shutil.copyfile(file_path, recording_file_path)
    print(f"🎙️ Recording saved to: {recording_file_path}")
    
    # return json end print
    print("\n🤖 Assistant Response:\n", response)

    return response

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/process-audio/")
async def process_audio(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp:
        shutil.copyfileobj(file.file, temp)
        temp_path = temp.name

    try:
        crm_json = process_audio_and_generate_crm_call(temp_path)
        return JSONResponse(content=crm_json)
    finally:
        os.remove(temp_path)

