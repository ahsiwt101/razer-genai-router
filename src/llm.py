"""LLM clients for the "llm" route (answering directly).

Plain text in, text out. We NEVER pass `tools`, `tool_config` or function declarations to the
model: routing is done entirely by our own router, which is the point of the assignment.

Two implementations with the same generate() method:
  GeminiClient  the real thing (needs GEMINI_API_KEY in .env)
  MockLLM       canned answer, no network: for tests and routing-only evaluation
"""
import os
import time

from dotenv import load_dotenv

# The system prompt is part of the routing design: when a live-data question reaches the LLM
# (router missed a tool, or a tool failed), the LLM must SAY it lacks current data instead of
# guessing. That's why config/router.yaml can treat "missed live tool" as a disclosed failure.
SYSTEM_PROMPT = (
    "You are a helpful, concise assistant. Answer in at most a few short paragraphs. "
    "You do not have access to real-time information such as today's weather, news, scores or prices: "
    "if a question needs live data, say clearly that you can't check current information, then give "
    "whatever general, timeless help you can. If asked to perform an action you cannot do (booking, "
    "sending messages, setting alarms), say so and suggest what the user could do instead."
)
DEFAULT_MODEL = "gemini-2.5-flash"


class LLMError(RuntimeError):
    pass


class GeminiClient:
    def __init__(self, model: str | None = None):
        from google import genai  # imported here so tests/mock runs don't need the SDK configured
        from google.genai import types

        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key or api_key == "your-key-here":
            raise LLMError("GEMINI_API_KEY is not set. Copy .env.example to .env and add your key.")
        self.model = model or os.getenv("GEMINI_MODEL") or DEFAULT_MODEL
        self._client = genai.Client(api_key=api_key)
        self._config = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=0.3)

    retries = 0  # how many rate-limit retries happened (the evaluation reports it: waits inflate latency)

    def generate(self, prompt: str) -> str:
        for attempt in range(4):
            try:
                response = self._client.models.generate_content(model=self.model, contents=prompt, config=self._config)
                return (response.text or "").strip()
            except Exception as e:  # network errors, quota, bad model name...
                rate_limited = "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)
                if rate_limited and attempt < 3:
                    self.retries += 1
                    time.sleep(2 ** (attempt + 1))  # 2s, 4s, 8s: back off politely, then try again
                    continue
                raise LLMError(f"Gemini call failed: {type(e).__name__}: {e}") from e


class MockLLM:
    model = "mock"

    def generate(self, prompt: str) -> str:
        first_line = prompt.strip().splitlines()[0]
        return f"[mock LLM] This is where the model would answer: '{first_line}'"
