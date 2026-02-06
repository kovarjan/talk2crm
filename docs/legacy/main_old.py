# from talk2text.chain import run_pipeline

# if __name__ == "__main__":
#     response = run_pipeline("assets/input.wav")
#     print("Response:", response)

from core.voice_assistant import transcribe_voice, get_api_call, clean_json_string
import json
import os

from core.pipelines.command_pipeline import process_voice_command

# For quick testing:
if __name__ == "__main__":

    # check if .env file exists
    if not os.path.exists(".env"):
        print("⚠️ .env file not found. Please create a .env file with the required environment variables.")
        exit(1)
    
    # audio_path = "assets/audio/untitled_exmaple2.mp3"
    # audio_path = "data/examples/audio/activity1_normal_grade1.mp3"
    audio_path = "data/examples/audio/activity3_normal_grade1.mp3"

    print("audio_path:", audio_path)
    
    # Process the voice command
    response = process_voice_command(audio_path)
    
    # return json end print
    print("\n🤖 Assistant Response:\n", response)



# if __name__ == "__main__":
#     mp3_path = "assets/untitled_exmaple2.mp3"
#     # text = transcribe_voice(mp3_path)
#     # print("🎙️ Transcribed:", text)

#     # text = "Vytvoř schůzku s firmou Acmark s.r.o. zítra v devět hodin."
#     text = "Vytvoř schůzku s Pavlem Novotným v úterý ve 12 hodin v Brně."
#     # text = "Schůzka s Alešem" 
#     # text = "Nová schůzka 19.5." 
#     print("\n🗣️ User said (manual override):", text)

#     response = get_api_call([{
#         "role": "user",
#         "content": text
#     }])
#     print("\n🤖 Assistant Response:\n", response)

#     try:
#         json_output = json.loads(clean_json_string(response))
#         print("\n✅ Parsed JSON:", json_output)

#         if json_output['mode'] == "confirm":
#             print("\n✅ Confirmation:", json_output['response'])
#             print("   > API Call:", json_output['api_call'])
#         elif json_output['mode'] == "clarify":
#             print("\n❓ Clarification needed:", json_output['response'])
#             print("   > API Call:", json_output['api_call'])

#             # Handle clarification input user input
#             # Clarification logic goes here
#             # For example, you can ask the user for more details
#             user_input = input("Please provide more details: ")

#             # user_corrected_text = "original message: " + text + "\n\nyour response: " + response + "\n\nusers clarification: " + user_input
#             user_corrected_text = [{
#                 "role": "user",
#                 "content": text
#             }, {
#                 "role": "assistant",
#                 "content": response
#             }, {
#                 "role": "user",
#                 "content": user_input
#             }]

#             response = get_api_call(user_corrected_text)
#             print("\n🤖 Assistant Response:\n", response)

#             try:
#                 json_output = json.loads(clean_json_string(response))
#                 print("\n✅ Parsed JSON:", json_output)
#                 print("\n🔄 Updated API Call:", json_output['api_call'])
        
#             except json.JSONDecodeError:
#                 print("\n⚠️ LLM Response not JSON – needs clarification END.")

#         else:
#             print("\n⚠️ Unexpected mode:", json_output.mode)
#             print("   > API Call:", json_output['api_call'])


#     except json.JSONDecodeError:
#         print("\n⚠️ LLM Response not JSON – needs clarification.")
