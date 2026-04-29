"""lensml.llm — LLM client wrapper and provider routing.

Per plan §5.5, four task categories with primary/secondary/tertiary providers:

    Task                            | Primary           | Secondary           | Tertiary
    ------------------------------- | ----------------- | ------------------- | -------------------
    Anomaly trigger classification  | Groq Llama 3.1 8B | Cerebras Llama 3.1  | Gemini 2.5 Flash-Lite
    Pathology diagnosis             | Gemini 2.5 Pro    | Cerebras GPT-OSS    | OpenRouter DeepSeek
    Code patch generation           | OR Qwen3-Coder    | Mistral Codestral   | Gemini 2.5 Pro
    Verification                    | Cerebras GPT-OSS  | Gemini 2.5 Pro      | -

LiteLLM normalizes most cross-provider tool-call schemas; per-provider quirks
(Gemini description-sensitivity, Groq strictness) handled in `client.py`.
"""
