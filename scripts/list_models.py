"""Sanity check: is the Gemini key valid, and which models can it use?

Run:  python scripts/list_models.py
"""
import os

from dotenv import load_dotenv
from google import genai

# Read KEY=value pairs from the .env file into environment variables.
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key or api_key == "your-key-here":
    raise SystemExit("No key found. Copy .env.example to .env and paste your key into .env.")

client = genai.Client(api_key=api_key)

# Only list models that can generate text ("generateContent"); skip embedding-only models etc.
print("Models your key can use for text generation:\n")
for model in client.models.list():
    if "generateContent" in (model.supported_actions or []):
        print(f"  {model.name}")

print(f"\nConfigured in .env: GEMINI_MODEL={os.getenv('GEMINI_MODEL')}")
