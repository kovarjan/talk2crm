#!/bin/bash

# ==========================================
# LLM Benchmark Runner (3 Runs per Model)
# ==========================================

# --- Configuration ---
RUNS=3
TEST_CMD="pytest tests/test_llm_evaluation.py -v -s"

# --- Endpoints & Keys ---
# Replace the placeholder keys before running the API tests
OLLAMA_URL="http://localhost:11434/v1"

OPENAI_URL="https://api.openai.com/v1"
OPENAI_KEY="YOUR_OPENAI_KEY"

GEMINI_URL="https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_KEY="YOUR_GEMINI_KEY"

# --- Helper Function ---
run_model() {
    local type=$1
    local model=$2
    local url=$3
    local key=$4

    echo "========================================================"
    echo "🧪 Testing Model: $model ($type)"
    echo "========================================================"

    for i in $(seq 1 $RUNS); do
        echo "⏳ Run $i of $RUNS for $model..."
        
        if [ "$type" == "base" ]; then
            # As requested, base Qwen 3 uses default env vars
            $TEST_CMD
            
        elif [ "$type" == "ollama" ]; then
            # Local Ollama test
            LLM_MODEL="$model" LLM_BASE_URL="$url" $TEST_CMD
            
        elif [ "$type" == "api" ]; then
            # Cloud API test (OpenAI or Gemini)
            LLM_MODEL="$model" LLM_BASE_URL="$url" LLM_API_KEY="$key" $TEST_CMD
        fi
        
        echo "✅ Finished run $i for $model"
        sleep 2 # Brief pause to allow ports/memory to flush
    done
    echo ""
}

echo "🚀 Starting Full LLM Thesis Benchmark Suite..."

# 1. Base Qwen 3 (Uses your default setup)
# run_model "base" "qwen3" "" ""

# 2. Local Ollama Models (RTX 3080/5080 Candidates)
OLLAMA_MODELS=(
    # "qwen3:14b"
    # "qwen3.6:latest"
    # "qwen2.5:14b-instruct"
    # "llama3.1:8b"
    # "llama3.2:latest"
    # "deepseek-r1:14b"
    "gemma4:latest"
    "gemma3:12b"
    "mistral:7b"
    "mistral-small3.1:24b"
    "mistral-nemo:12b"
)

for model in "${OLLAMA_MODELS[@]}"; do
    run_model "ollama" "$model" "$OLLAMA_URL" ""
done

# 3. OpenAI Models
OPENAI_MODELS=(
    "gpt-5.4"
    "gpt-5.4-mini"
    "gpt-5.4-nano"
    "gpt-4o-mini"
)

for model in "${OPENAI_MODELS[@]}"; do
    run_model "api" "$model" "$OPENAI_URL" "$OPENAI_KEY"
done

# 4. Gemini Models (Using OpenAI compatibility endpoint)
GEMINI_MODELS=(
    "gemini-2.5-flash"
    "gemini-2.5-flash-lite"
    "gemini-1.5-pro"
)

for model in "${GEMINI_MODELS[@]}"; do
    run_model "api" "$model" "$GEMINI_URL" "$GEMINI_KEY"
done

echo "🎉 All benchmark runs completed successfully!"