import json
from core.services.llm import query_llm

def SttCorrectorAgent(prompt: str) -> str:
    """
    This function takes a prompt transcribed from speech and returns corrected text.
    It uses a LLM to process the prompt and correct any errors from STT.
    """

    systemPrompt = (
        "You are a voice assistant agent. Correct the user's spoken request.\n"
        "Make sure to correct any errors in the transcription and relay the meaning.\n"
        "Return the corrected text only."
    )

    print(f"🤖 [SttCorrectorAgent] Final Prompt: {prompt}")

    response = query_llm(prompt, temperature=0.4, isTemplate=True, systemPrompt=systemPrompt, replaceSystemPrompt=True, returnJson=False)
    print(f"🤖 [SttCorrectorAgent] Corrected: {response}")

    return response
